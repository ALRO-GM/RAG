// competition.html, the page's address before #295, now only sends the
// browser to info.html, keeping the section in the address (#coste,
// #tareas…), so links shared before the rename keep working. The
// <meta http-equiv="refresh"> of that page does the same without the
// section when scripts are off.
location.replace(`info.html${location.hash}`);
