use std::time::{Instant, UNIX_EPOCH};

use cudarc::{
    jit::compile_ptx,
    prelude::{CudaDeviceBuilder, DriverError, LaunchAsync, LaunchConfig},
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
fn add_one(mut v: Vec<u32>) -> Vec<u32> {
    let mut i = v.len();
    let mut overflowed = true;
    while overflowed {
        i -= 1;
        let (new, just_overflowed) = v[i].overflowing_add(1);
        v[i] = new;
        overflowed = just_overflowed;
    }
    v
}
unsafe fn _main(block_size: u32, grid_size: u32) -> Result<f64, DriverError> {
    let cuda = CudaDeviceBuilder::new(0)
        .with_ptx(
            compile_ptx(include_str!("./../sha256_kernel.cu")).unwrap(),
            "sha256",
            &["sha256"],
        )
        .build()
        .unwrap();
    let sha256 = cuda.get_func("sha256", "sha256").unwrap();
    // block from https://en.bitcoin.it/wiki/Block_hashing_algorithm
    let block = Block {
        version: 0x01000000,
        prev: [
            0x81cd02ab, 0x7e569e8b, 0xcd9317e2, 0xfe99f2de, 0x44d49ab2, 0xb8851ba4, 0xa3080000,
            0x00000000,
        ],
        merkle: [
            0xe320b6c2, 0xfffc8d75, 0x0423db8b, 0x1eb942ae, 0x710e951e, 0xd797f7af, 0xfc8892b0,
            0xf1fc122b,
        ],
        time: UNIX_EPOCH.elapsed().unwrap().as_secs() as u32,
        bits: 0xf2b9441a,
        nonce: 0,
    };
    let block_slice = any_as_u32_slice(&block);
    let first_part = &block_slice[..16];
    let initial = vec![
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
    assert_eq!(&helper[..16], first_part);
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

    let mut worker = initial.clone();

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
    for i in 0..8 {
        worker[i] = worker[i].wrapping_add(initial[i]);
    }

    let mut to_hash = [0u32; 16];
    to_hash[..4].copy_from_slice(&block_slice[16..]);
    to_hash[4] = 1 << (u32::BITS - 1);
    to_hash[16 - 1] = 80 * 8;
    let hash_io = cuda.take_async(to_hash.to_vec()).unwrap();
    let mut worker = cuda.take_async(worker).unwrap();
    let initial = cuda.take_async(initial).unwrap();
    let target = cuda.take_async(
        add_one(vec![
            0x00000000u32,
            0x00000000,
            0xFFFFFFFF,
            0xFFFFFFFF,
            0xFFFFFFFF,
            0xFFFFFFFF,
            0xFFFFFFFF,
            0xFFFFFFFF,
        ])
        .into_iter()
        .rev()
        .map(|b| b.to_le_bytes())
        .collect(),
    )?;
    let mut finished_counter = cuda.take_async(vec![0]).unwrap();
    let start = Instant::now();
    sha256.launch_async(
        LaunchConfig {
            block_dim: (block_size, 1, 1),
            grid_dim: (grid_size, 1, 1),
            shared_mem_bytes: 0,
        },
        (
            &hash_io,
            &mut worker,
            &target,
            &initial,
            &mut finished_counter,
        ),
    )?;
    cuda.synchronize()?;
    let elapsed = start.elapsed().as_secs_f64();
    // println!("Elapsed: {elapsed:?}s");
    // let output = cuda.sync_release(worker).unwrap();
    // println!(
    //     "0x{:0>28}00000000 with nonce 0x{:x}",
    //     output[..7]
    //         .iter()
    //         .map(|n| format!("{n:0>4x}"))
    //         .collect::<Vec<_>>()
    //         .join(""),
    //     output[7]
    // );
    // println!(
    //     "Finished threads: {}",
    //     cuda.sync_release(finished_counter).unwrap()[0]
    // );
    Ok(elapsed)
}

fn main() {
    // for block_size in [4, 8, 16, 32, 64, 128, 256, 512, 1024] {
    //     for grid_size in [4, 8, 16, 32, 64, 128, 256, 512, 1024] {
    //         if block_size * grid_size < 4096 {continue;}
    //         print!("{block_size}|{grid_size}: ");
    //         std::io::stdout().flush().unwrap();
    //         unsafe {
    //             match _main(block_size, grid_size) {
    //                 Ok(elapsed) => println!("{}s", elapsed),

    //                 Err(e) => println!("Error {e}"),
    //             };
    //         }
    //     }
    // }
    //
    // 256/16 is most efficient, 33.8s for 2^32 hashes on a
    // NVIDIA GeForce GTX 1050 ti;
    // 2^32H / 33.8s = 127070038.343 H/s = 127.07 MH/s,
    // about 10-100x faster than the hashrate stated online
    unsafe {
        println!("Elapsed: {}s", _main(256, 16).expect("No nonce found."));
    }
}

// Results:
//     {
//         "blockSize": "4",
//         "gridSize": "1024",
//         "time": 130.8206026
//     },
//     {
//         "blockSize": "8",
//         "gridSize": "512",
//         "time": 63.9826427
//     },
//     {
//         "blockSize": "8",
//         "gridSize": "1024",
//         "time": 62.5828092
//     },
//     {
//         "blockSize": "16",
//         "gridSize": "256",
//         "time": 49.8715968
//     },
//     {
//         "blockSize": "16",
//         "gridSize": "512",
//         "time": 39.6408215
//     },
//     {
//         "blockSize": "16",
//         "gridSize": "1024",
//         "time": 40.7138623
//     },
//     {
//         "blockSize": "32",
//         "gridSize": "128",
//         "time": 32.7539834
//     },
//     {
//         "blockSize": "32",
//         "gridSize": "256",
//         "time": 37.5859184
//     },
//     {
//         "blockSize": "32",
//         "gridSize": "512",
//         "time": 36.0540847
//     },
//     {
//         "blockSize": "32",
//         "gridSize": "1024",
//         "time": 37.0893008
//     },
//     {
//         "blockSize": "64",
//         "gridSize": "64",
//         "time": 32.5774772
//     },
//     {
//         "blockSize": "64",
//         "gridSize": "128",
//         "time": 37.9311008
//     },
//     {
//         "blockSize": "64",
//         "gridSize": "256",
//         "time": 35.9099194
//     },
//     {
//         "blockSize": "64",
//         "gridSize": "512",
//         "time": 37.292432
//     },
//     {
//         "blockSize": "64",
//         "gridSize": "1024",
//         "time": 36.9034332
//     },
//     {
//         "blockSize": "128",
//         "gridSize": "32",
//         "time": 32.5754998
//     },
//     {
//         "blockSize": "128",
//         "gridSize": "64",
//         "time": 37.9085129
//     },
//     {
//         "blockSize": "128",
//         "gridSize": "128",
//         "time": 35.941195
//     },
//     {
//         "blockSize": "128",
//         "gridSize": "256",
//         "time": 37.1024726
//     },
//     {
//         "blockSize": "128",
//         "gridSize": "512",
//         "time": 36.860343
//     },
//     {
//         "blockSize": "128",
//         "gridSize": "1024",
//         "time": 37.1873604
//     },
//     {
//         "blockSize": "256",
//         "gridSize": "16",
//         "time": 32.2261883
//     },
//     {
//         "blockSize": "256",
//         "gridSize": "32",
//         "time": 37.8734151
//     },
//     {
//         "blockSize": "256",
//         "gridSize": "64",
//         "time": 35.9127096
//     },
//     {
//         "blockSize": "256",
//         "gridSize": "128",
//         "time": 36.8671456
//     },
//     {
//         "blockSize": "256",
//         "gridSize": "256",
//         "time": 36.8247113
//     },
//     {
//         "blockSize": "256",
//         "gridSize": "512",
//         "time": 37.0394992
//     },
//     {
//         "blockSize": "256",
//         "gridSize": "1024",
//         "time": 38.3367265
//     },
//     {
//         "blockSize": "512",
//         "gridSize": "8",
//         "time": 33.9912728
//     },
//     {
//         "blockSize": "512",
//         "gridSize": "16",
//         "time": 38.3980109
//     },
//     {
//         "blockSize": "512",
//         "gridSize": "32",
//         "time": 37.0195985
//     },
//     {
//         "blockSize": "512",
//         "gridSize": "64",
//         "time": 38.7311395
//     },
//     {
//         "blockSize": "512",
//         "gridSize": "128",
//         "time": 40.8564679
//     },
//     {
//         "blockSize": "512",
//         "gridSize": "256",
//         "time": 125.7235859
//     },
//     {
//         "blockSize": "512",
//         "gridSize": "512",
//         "time": 100.2657007
//     },
//     {
//         "blockSize": "512",
//         "gridSize": "1024",
//         "time": 99.4688822
//     },
//     {
//         "blockSize": "1024",
//         "gridSize": "4",
//         "time": 61.1782466
//     },
//     {
//         "blockSize": "1024",
//         "gridSize": "8",
//         "time": 73.7160563
//     },
//     {
//         "blockSize": "1024",
//         "gridSize": "16",
//         "time": 125.4535661
//     },
//     {
//         "blockSize": "1024",
//         "gridSize": "32",
//         "time": 109.4788913
//     },
//     {
//         "blockSize": "1024",
//         "gridSize": "64",
//         "time": 103.3825721
//     },
//     {
//         "blockSize": "1024",
//         "gridSize": "128",
//         "time": 113.6010112
//     },
//     {
//         "blockSize": "1024",
//         "gridSize": "256",
//         "time": 109.2072914
//     },
//     {
//         "blockSize": "1024",
//         "gridSize": "512",
//         "time": 99.2597946
//     },
//     {
//         "blockSize": "1024",
//         "gridSize": "1024",
//         "time": 90.2409356
//     }
// ]
