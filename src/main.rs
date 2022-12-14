use std::{mem::MaybeUninit, rc::Rc};

use cudarc::{
    jit::compile_ptx,
    prelude::{CudaDeviceBuilder, LaunchConfig, LaunchCudaFunction},
};

const K: [u32; 64] = [
    0x428a2f98u32,
    0x71374491,
    0xb5c0fbcf,
    0xe9b5dba5,
    0x3956c25b,
    0x59f111f1,
    0x923f82a4,
    0xab1c5ed5,
    0xd807aa98,
    0x12835b01,
    0x243185be,
    0x550c7dc3,
    0x72be5d74,
    0x80deb1fe,
    0x9bdc06a7,
    0xc19bf174,
    0xe49b69c1,
    0xefbe4786,
    0x0fc19dc6,
    0x240ca1cc,
    0x2de92c6f,
    0x4a7484aa,
    0x5cb0a9dc,
    0x76f988da,
    0x983e5152,
    0xa831c66d,
    0xb00327c8,
    0xbf597fc7,
    0xc6e00bf3,
    0xd5a79147,
    0x06ca6351,
    0x14292967,
    0x27b70a85,
    0x2e1b2138,
    0x4d2c6dfc,
    0x53380d13,
    0x650a7354,
    0x766a0abb,
    0x81c2c92e,
    0x92722c85,
    0xa2bfe8a1,
    0xa81a664b,
    0xc24b8b70,
    0xc76c51a3,
    0xd192e819,
    0xd6990624,
    0xf40e3585,
    0x106aa070,
    0x19a4c116,
    0x1e376c08,
    0x2748774c,
    0x34b0bcb5,
    0x391c0cb3,
    0x4ed8aa4a,
    0x5b9cca4f,
    0x682e6ff3,
    0x748f82ee,
    0x78a5636f,
    0x84c87814,
    0x8cc70208,
    0x90befffa,
    0xa4506ceb,
    0xbef9a3f7,
    0xc67178f2,
];

fn main() {
    unsafe {
        _main();
    }
}

#[repr(C)]
#[repr(packed)]
struct Block {
    version: u32,
    prev: [u32; 8],
    merkle: [u32; 8],
    time: u32,
    bits: u32,
    nonce: u32,
}

const unsafe fn any_as_u32_slice<T: Sized>(p: &T) -> &[u32] {
    ::std::slice::from_raw_parts(
        (p as *const T) as *const u32,
        ::std::mem::size_of::<T>() / ::std::mem::size_of::<u32>(),
    )
}

unsafe fn _main() {
    let cuda = CudaDeviceBuilder::new(0)
        .with_ptx(
            "sha256",
            compile_ptx(include_str!("./../sha256_kernel.cu")).unwrap(),
            &["sha256"],
        )
        .build()
        .unwrap();
    let sha256 = cuda.get_module("sha256").unwrap().get_fn("sha256").unwrap();
    let block = any_as_u32_slice(&Block {
        version: 0x01000000,
        prev: [
            0x81cd02ab, 0x7e569e8b, 0xcd9317e2, 0xfe99f2de, 0x44d49ab2, 0xb8851ba4, 0xa3080000,
            0x00000000,
        ],
        merkle: [
            0xe320b6c2, 0xfffc8d75, 0x0423db8b, 0x1eb942ae, 0x710e951e, 0xd797f7af, 0xfc8892b0,
            0xf1fc122b,
        ],
        time: 0xc7f5d74d,
        bits: 0xf2b9441a,
        nonce: 0,
    });
    let mut data =
        *b"abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ__abcdefghijklmnopqrstuvwxyz";
    let t = (&mut *data.as_mut_ptr().cast::<[u32; 20]>()).as_mut_slice();
    for chunk in t.iter_mut() {
        *chunk = chunk.to_be();
    }
    println!(
        "{}",
        data.iter().map(|c| char::from(*c)).collect::<String>()
    );
    let block = &*t;
    let first_part = &block[..16];
    let initial = [
        0x6a09e667u32,
        0xbb67ae85,
        0x3c6ef372,
        0xa54ff53a,
        0x510e527f,
        0x9b05688c,
        0x1f83d9ab,
        0x5be0cd19,
    ];
    let mut helper = [0u32; 64];
    helper[..16].copy_from_slice(first_part);
    assert_eq!(&helper[..16], &first_part[..]);
    for i in 16..64 {
        let s0 = helper[i - 15].rotate_right(7)
            ^ helper[i - 15].rotate_right(18)
            ^ helper[i - 15].wrapping_shr(3);
        let s1 = helper[i - 2].rotate_right(17)
            ^ helper[i - 2].rotate_right(19)
            ^ helper[i - 2].wrapping_shr(10);
        helper[i] = helper[i - 16]
            .wrapping_add(s0)
            .wrapping_add(helper[i - 7])
            .wrapping_add(s1);
    }

    let mut worker = initial;

    for i in 0..64 {
        let s1 =
            worker[4].rotate_right(6) ^ worker[4].rotate_right(11) ^ worker[4].rotate_right(25);
        let ch = (worker[4] & worker[5]) ^ ((!worker[4]) & worker[6]);
        let t1 = worker[7]
            .wrapping_add(s1)
            .wrapping_add(ch)
            .wrapping_add(K[i])
            .wrapping_add(helper[i]);
        let s0 =
            worker[0].rotate_right(2) ^ worker[0].rotate_right(13) ^ worker[0].rotate_right(22);
        let maj = (worker[0] & worker[1]) ^ (worker[0] & worker[2]) ^ (worker[1] & worker[2]);
        let t2 = s0.wrapping_add(maj);

        worker[7] = worker[6];
        worker[6] = worker[5];
        worker[5] = worker[4];
        worker[4] = worker[3].wrapping_add(t1);
        worker[3] = worker[2];
        worker[2] = worker[1];
        worker[1] = worker[0];
        worker[0] = t1.wrapping_add(t2);
    }
    worker[7] = worker[7].wrapping_add(initial[7]);
    worker[6] = worker[6].wrapping_add(initial[6]);
    worker[5] = worker[5].wrapping_add(initial[5]);
    worker[4] = worker[4].wrapping_add(initial[4]);
    worker[3] = worker[3].wrapping_add(initial[3]);
    worker[2] = worker[2].wrapping_add(initial[2]);
    worker[1] = worker[1].wrapping_add(initial[1]);
    worker[0] = worker[0].wrapping_add(initial[0]);

    println!("{worker:x?}");

    let mut to_hash = [0u32; 64];
    to_hash[..4].copy_from_slice(&t[16..]);
    to_hash[4] = 1 << (u32::BITS - 1);
    to_hash[16 - 1] = 80 * 8;
    let mut hash_io = cuda.take(Rc::new(to_hash)).unwrap();
    let mut worker = cuda.take(Rc::new(worker)).unwrap();
    let k = cuda.take(Rc::new(K)).unwrap();
    cuda.launch_cuda_function(
        sha256,
        LaunchConfig::for_num_elems(1),
        (&mut hash_io, &mut worker, &k),
    )
    .unwrap();
    let output = &hash_io.into_host().unwrap()[..8];
    println!(
        "0x{}",
        output
            .iter()
            .map(|n| format!("{n:0>4x}"))
            .collect::<Vec<_>>()
            .join("")
    );
}
