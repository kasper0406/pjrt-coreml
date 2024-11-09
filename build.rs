use std::env;
use std::path::PathBuf;

fn main() {
    // PJRT C API bindings
    let bindings = bindgen::Builder::default()
        .header("third_party/xla/pjrt_c_api.h")
        .parse_callbacks(Box::new(bindgen::CargoCallbacks::new()))
        .generate()
        .expect("Unable to generate bindings");

    // Write the bindings to the $OUT_DIR/bindings.rs file.
    let out_path = PathBuf::from(env::var("OUT_DIR").unwrap());
    bindings
        .write_to_file(out_path.join("bindings.rs"))
        .expect("Couldn't write bindings!");

    // Swift CoreML library
    println!("cargo::rustc-link-search=/Volumes/git/ml/pjrt-coreml/swift/");
    println!("cargo::rustc-link-search=/usr/lib/swift/");
    println!("cargo::rustc-link-lib=static=swift_coreml");
    println!("cargo::rustc-link-lib=swiftCore");

    // Swift bindings
    let path = std::path::PathBuf::from(".");
    let mut b = autocxx_build::Builder::new("src/coreml.rs", &[&path]).build().unwrap();
    b.flag_if_supported("-std=c++20").compile("autocxx_swift_coreml");

    println!("cargo:rerun-if-changed=src/lib.rs");
    println!("cargo:rerun-if-changed=src/coreml.rs");
    println!("cargo:rerun-if-changed=swift/libswift_coreml.a");
}
