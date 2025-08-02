// build.rs (in the crate root)
use std::{env, path::PathBuf};

fn main() {
    // Absolute path to <crate>/native/lib
    let root = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());
    let lib_dir = root.join("..").join("cuda_octree").join("build");

    println!(
        "cargo:rerun-if-changed={}/libCudaOctree.a",
        lib_dir.display()
    );

    // ① Where rustc should search for libraries
    println!("cargo:rustc-link-search=native={}", lib_dir.display());

    println!("cargo:rustc-link-lib=static=CudaOctree");

    println!("cargo:rustc-link-search=native=/usr/local/cuda/lib64");

    println!("cargo:rustc-link-lib=dylib=stdc++");

    println!("cargo:rustc-link-lib=dylib=cudart"); // CUDA Runtime

    println!("cargo:rustc-link-lib=dylib=cuda"); // CUDA Driver API

    println!("cargo:rustc-link-arg=-Wl,-rpath,/usr/local/cuda/lib64");

    // ③ Optional: generate bindings with bindgen instead of hand-writing
    /*
    let bindings = bindgen::Builder::default()
        .header(root.join("native").join("include").join("foo.h").to_string_lossy())
        .generate()
        .expect("Unable to generate bindings");
    bindings
        .write_to_file(root.join("src").join("bindings.rs"))
        .expect("Couldn't write bindings!");
    */
}
