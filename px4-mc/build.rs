use std::env;
use std::path::PathBuf;

fn main() {
    // Allow users to point at an existing PX4-Autopilot tree
    // via PX4_SRC_DIR env var. This gets forwarded to CMake.
    let mut cfg = cmake::Config::new("cpp/px4_mc");

    if let Ok(px4_src) = env::var("PX4_SRC_DIR") {
        cfg.define("PX4_SRC_DIR", &px4_src);
    }

    // Build the C++ static library with CMake
    let dst = cfg.build();

    // Link the produced static lib
    let mut libdir = PathBuf::from(dst);
    libdir.push("lib");
    println!("cargo:rustc-link-search=native={}", libdir.display());
    println!("cargo:rustc-link-lib=static=px4_mc");

    // Link the C++ stdlib depending on target
    let target = env::var("TARGET").unwrap_or_default();
    if target.contains("apple-darwin") {
        println!("cargo:rustc-link-lib=c++");
    } else if target.contains("windows") {
        // MSVC links C++ runtime automatically
    } else {
        println!("cargo:rustc-link-lib=stdc++");
    }
}

