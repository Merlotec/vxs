use std::env;
use std::fs;
use std::io::{BufRead, BufReader, Read, Write};
use std::net::TcpStream;
use std::time::Duration;

fn print_usage() {
    eprintln!("Usage: voxelsim-controller [--addr host:port] <command> [args]\n");
    eprintln!("Commands:");
    eprintln!("  takeoff <alt_m>");
    eprintln!("  engage \"<python_one_line_script>\"");
    eprintln!("  engage-file <path>   (reads file; joins lines with spaces)");
    eprintln!("  engage-manual");
    eprintln!("  action <urgency> <yaw> <moves_csv>   (e.g., 1.0 0.0 1,1,5)");
    eprintln!("  disengage");
    eprintln!("  land");
    eprintln!("  kill");
    eprintln!("  quit");
    eprintln!("  repl                  (interactive mode; default)");
    eprintln!("");
    eprintln!("Embedding: use $path or $\"path with spaces\" to inline file contents; escape literal $ with $$");
}

fn connect(addr: &str) -> std::io::Result<TcpStream> {
    let mut stream = TcpStream::connect(addr)?;
    stream.set_read_timeout(Some(Duration::from_millis(200)))?;
    Ok(stream)
}

fn send_and_print(mut stream: TcpStream, line: &str) -> std::io::Result<()> {
    let mut to_send = expand_embeds(line);
    if !to_send.ends_with('\n') {
        to_send.push('\n');
    }
    stream.write_all(to_send.as_bytes())?;
    stream.flush()?;

    // Read until we get at least some response or we hit a short deadline.
    let deadline = std::time::Instant::now() + Duration::from_millis(1200);
    let mut got_any = false;
    let mut buf = [0u8; 4096];
    loop {
        match stream.read(&mut buf) {
            Ok(0) => {
                if got_any || std::time::Instant::now() > deadline { break; }
            }
            Ok(n) => {
                if n > 0 { got_any = true; }
                let s = String::from_utf8_lossy(&buf[..n]);
                print!("{}", s);
                // Keep draining until timeout after first chunk
            }
            Err(ref e) if e.kind() == std::io::ErrorKind::WouldBlock || e.kind() == std::io::ErrorKind::TimedOut => {
                if got_any || std::time::Instant::now() > deadline { break; }
            }
            Err(e) => return Err(e),
        }
    }
    Ok(())
}

fn run_repl(addr: &str) -> std::io::Result<()> {
    let mut stream = connect(addr)?;
    // Switch to blocking reads; spawn a reader thread to print responses as they arrive
    stream.set_read_timeout(None)?;

    let reader_stream = stream.try_clone()?;
    std::thread::spawn(move || {
        let mut reader = BufReader::new(reader_stream);
        loop {
            let mut line = String::new();
            match reader.read_line(&mut line) {
                Ok(0) => break, // disconnected
                Ok(_) => {
                    if !line.is_empty() {
                        print!("{}", line);
                        let _ = std::io::Write::flush(&mut std::io::stdout());
                    }
                }
                Err(_) => break,
            }
        }
    });

    // REPL loop: read stdin lines and forward
    let stdin = std::io::stdin();
    let mut input = String::new();
    loop {
        input.clear();
        print!("> ");
        let _ = std::io::Write::flush(&mut std::io::stdout());
        if stdin.read_line(&mut input)? == 0 { break; }
        if input.trim().is_empty() { continue; }
        let expanded = expand_embeds(&input);
        stream.write_all(expanded.as_bytes())?;
        stream.flush()?;
        if input.trim().eq_ignore_ascii_case("quit") { break; }
    }
    Ok(())
}

// Replace $"path with spaces" or $path tokens with the file contents inline.
// Newlines in the file are converted to spaces to keep a single-line protocol.
fn expand_embeds(input: &str) -> String {
    let mut out = String::with_capacity(input.len());
    let bytes = input.as_bytes();
    let mut i = 0;
    while i < bytes.len() {
        if bytes[i] == b'$' {
            // Escape sequence: $$ -> literal $
            if i + 1 < bytes.len() && bytes[i + 1] == b'$' {
                out.push('$');
                i += 2;
                continue;
            }
            let start = i;
            i += 1;
            if i < bytes.len() && bytes[i] == b'"' {
                // Quoted path: $"..."
                i += 1; // skip opening quote
                let path_start = i;
                while i < bytes.len() && bytes[i] != b'"' { i += 1; }
                if i >= bytes.len() { // no closing quote; treat as literal
                    out.push_str(&input[start..]);
                    break;
                }
                let path = &input[path_start..i];
                i += 1; // skip closing quote
                match fs::read_to_string(path) {
                    Ok(content) => {
                        let one_line = content.replace('\n', " ");
                        out.push_str(&one_line);
                    }
                    Err(e) => {
                        eprintln!("embed error reading \"{}\": {}", path, e);
                        // fall back to literal token
                        out.push_str(&input[start..i]);
                    }
                }
            } else {
                // Unquoted path: read until whitespace
                let path_start = i;
                while i < bytes.len() && !bytes[i].is_ascii_whitespace() { i += 1; }
                if path_start == i { // just a lone '$'
                    out.push('$');
                    continue;
                }
                let path = &input[path_start..i];
                match fs::read_to_string(path) {
                    Ok(content) => {
                        let one_line = content.replace('\n', " ");
                        out.push_str(&one_line);
                    }
                    Err(e) => {
                        eprintln!("embed error reading {}: {}", path, e);
                        // fall back to literal token
                        out.push('$');
                        out.push_str(path);
                    }
                }
            }
        } else {
            out.push(bytes[i] as char);
            i += 1;
        }
    }
    out
}

fn main() -> std::io::Result<()> {
    let mut args = env::args().skip(1).collect::<Vec<_>>();
    let mut addr =
        env::var("VOXELSIM_DAEMON_ADDR").unwrap_or_else(|_| "127.0.0.1:7000".to_string());

    // Optional --addr override
    if args.len() >= 2 && args[0] == "--addr" {
        addr = args[1].clone();
        args.drain(0..2);
    }

    if args.is_empty() {
        println!("Listening for control inputs...");
        return run_repl(&addr);
    }

    let cmd = args.remove(0);
    match cmd.as_str() {
        "takeoff" if args.len() == 1 => {
            let alt = &args[0];
            let stream = connect(&addr)?;
            send_and_print(stream, &format!("TAKEOFF {}", alt))
        }
        "engage" if args.len() == 1 => {
            let script = &args[0];
            let stream = connect(&addr)?;
            send_and_print(stream, &format!("ENGAGE {}", script))
        }
        "engage-file" if args.len() == 1 => {
            let path = &args[0];
            let content = fs::read_to_string(path)?;
            // Server expects a single line; best-effort join
            let one_line = content.replace('\n', " ");
            let stream = connect(&addr)?;
            send_and_print(stream, &format!("ENGAGE {}", one_line))
        }
        "engage-manual" => {
            let stream = connect(&addr)?;
            send_and_print(stream, "ENGAGE_MANUAL")
        }
        "action" if args.len() == 3 => {
            let urg = &args[0];
            let yaw = &args[1];
            let moves = &args[2];
            let stream = connect(&addr)?;
            // Manual protocol: ACTION <urgency> <yaw> <moves_csv>
            send_and_print(stream, &format!("ACTION {} {} {}", urg, yaw, moves))
        }
        "disengage" => {
            let stream = connect(&addr)?;
            send_and_print(stream, "DISENGAGE")
        }
        "land" => {
            let stream = connect(&addr)?;
            send_and_print(stream, "LAND")
        }
        "kill" => {
            let stream = connect(&addr)?;
            send_and_print(stream, "KILL")
        }
        "quit" => {
            let stream = connect(&addr)?;
            send_and_print(stream, "QUIT")
        }
        "repl" => run_repl(&addr),
        _ => {
            print_usage();
            std::process::exit(2);
        }
    }
}
