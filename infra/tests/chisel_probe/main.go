// Command chisel_probe is a hand-made chisel client for the docker proof
// (infra/tests/docker_proof.py); it is never deployed.
//
// It speaks chisel's protocol (SSH over a WebSocket, subprotocol
// "chisel-v3") the way a modified client could: it authenticates with
// the credential in AUTH, declares one remote in the "config" request,
// and then opens one SSH channel of type "chisel" per target given on
// the command line, whatever it declared. The stock client only opens
// channels for the remotes it declared; this one asks the server to
// dial arbitrary addresses from inside its container.
//
// Usage: chisel_probe <ws-url> <declared remote> <target>...
// The declared remote is R:<host>:<port>:<host>:<port>. It prints one
// line per step:
//
//	config accepted | config refused: <message>
//	channel <target> opened: <first line of the answer to an HTTP GET>
//	channel <target> refused: <reason>: <message>
package main

import (
	"bufio"
	"encoding/json"
	"fmt"
	"net/http"
	"os"
	"strings"
	"time"

	"github.com/gorilla/websocket"
	"golang.org/x/crypto/ssh"
)

// remote mirrors chisel's settings.Remote, as its config request
// carries it (JSON of the exported fields).
type remote struct {
	LocalHost, LocalPort, LocalProto    string
	RemoteHost, RemotePort, RemoteProto string
	Socks, Reverse, Stdio               bool
}

// config mirrors chisel's settings.Config.
type config struct {
	Version string
	Remotes []remote
}

// wsConn turns binary WebSocket messages into a byte stream.
type wsConn struct {
	*websocket.Conn
	buff []byte
}

func (c *wsConn) Read(dst []byte) (int, error) {
	if len(c.buff) == 0 {
		_, msg, err := c.Conn.ReadMessage()
		if err != nil {
			return 0, err
		}
		c.buff = msg
	}
	n := copy(dst, c.buff)
	c.buff = c.buff[n:]
	return n, nil
}

func (c *wsConn) Write(b []byte) (int, error) {
	if err := c.Conn.WriteMessage(websocket.BinaryMessage, b); err != nil {
		return 0, err
	}
	return len(b), nil
}

func (c *wsConn) SetDeadline(t time.Time) error {
	if err := c.Conn.SetReadDeadline(t); err != nil {
		return err
	}
	return c.Conn.SetWriteDeadline(t)
}

func declared(spec string) (remote, error) {
	parts := strings.Split(spec, ":")
	if len(parts) != 5 || parts[0] != "R" {
		return remote{}, fmt.Errorf("not R:host:port:host:port: %s", spec)
	}
	return remote{
		LocalHost: parts[1], LocalPort: parts[2], LocalProto: "tcp",
		RemoteHost: parts[3], RemotePort: parts[4], RemoteProto: "tcp",
		Reverse: true,
	}, nil
}

func main() {
	if len(os.Args) < 3 {
		fmt.Fprintln(os.Stderr, "usage: chisel_probe <ws-url> <remote> <target>...")
		os.Exit(2)
	}
	user, pass, _ := strings.Cut(os.Getenv("AUTH"), ":")
	r, err := declared(os.Args[2])
	if err != nil {
		fmt.Println("error:", err)
		os.Exit(2)
	}
	dialer := websocket.Dialer{
		HandshakeTimeout: 15 * time.Second,
		Subprotocols:     []string{"chisel-v3"},
	}
	ws, _, err := dialer.Dial(os.Args[1], http.Header{})
	if err != nil {
		fmt.Println("error: websocket:", err)
		os.Exit(1)
	}
	conn := &wsConn{Conn: ws}
	sshConn, chans, reqs, err := ssh.NewClientConn(conn, "", &ssh.ClientConfig{
		User:            user,
		Auth:            []ssh.AuthMethod{ssh.Password(pass)},
		ClientVersion:   "SSH-chisel-v3-client",
		HostKeyCallback: ssh.InsecureIgnoreHostKey(),
		Timeout:         15 * time.Second,
	})
	if err != nil {
		fmt.Println("error: ssh:", err)
		os.Exit(1)
	}
	go ssh.DiscardRequests(reqs)
	go func() {
		// Connections to the reverse port: not wanted here.
		for ch := range chans {
			ch.Reject(ssh.Prohibited, "probe")
		}
	}()
	payload, _ := json.Marshal(config{Version: "1.12.0", Remotes: []remote{r}})
	ok, reply, err := sshConn.SendRequest("config", true, payload)
	if err != nil {
		fmt.Println("error: config:", err)
		os.Exit(1)
	}
	if !ok {
		fmt.Printf("config refused: %s\n", reply)
		os.Exit(1)
	}
	fmt.Println("config accepted")
	for _, target := range os.Args[3:] {
		ch, chReqs, err := sshConn.OpenChannel("chisel", []byte(target))
		if err != nil {
			if e, isOpen := err.(*ssh.OpenChannelError); isOpen {
				fmt.Printf("channel %s refused: %s: %s\n", target, e.Reason, e.Message)
			} else {
				fmt.Printf("channel %s refused: %s\n", target, err)
			}
			continue
		}
		go ssh.DiscardRequests(chReqs)
		fmt.Fprintf(ch, "GET / HTTP/1.0\r\nHost: probe\r\n\r\n")
		line := make(chan string, 1)
		go func() {
			text, _ := bufio.NewReader(ch).ReadString('\n')
			line <- strings.TrimSpace(text)
		}()
		select {
		case text := <-line:
			fmt.Printf("channel %s opened: %q\n", target, text)
		case <-time.After(5 * time.Second):
			fmt.Printf("channel %s opened: (no answer in 5 s)\n", target)
		}
		ch.Close()
	}
	sshConn.Close()
}
