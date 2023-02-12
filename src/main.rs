use std::{
    iter::once,
    time::{Duration, Instant, UNIX_EPOCH},
};

use bitcoin::{
    hashes::hex::FromHex, secp256k1::ThirtyTwoByteHash, util::uint::Uint256, BlockHash,
    BlockHeader, TxMerkleNode,
};
use cudarc::{
    jit::compile_ptx,
    prelude::{CudaDeviceBuilder, DriverError, LaunchAsync, LaunchConfig},
};
use hex_literal::hex;
use itertools::Itertools;

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

fn block_to_u32_slice(header: &BlockHeader) -> Vec<u32> {
    once((header.version as u32).swap_bytes())
        .chain(
            header
                .prev_blockhash
                .as_hash()
                .into_32()
                .chunks(4)
                .map(|c| u32::from_be_bytes(c.try_into().unwrap()))
                .collect_vec(),
        )
        .chain(
            header
                .merkle_root
                .as_hash()
                .into_32()
                .chunks(4)
                .map(|c| u32::from_be_bytes(c.try_into().unwrap()))
                .collect_vec(),
        )
        .chain(once(header.time.swap_bytes()))
        .chain(once(header.bits.swap_bytes()))
        .chain(once(header.nonce.swap_bytes()))
        .collect_vec()
}
unsafe fn _main(
    block_size: u32,
    grid_size: u32,
    avg_hashes_per_second: u64,
) -> Result<(f64, Option<BlockHeader>), DriverError> {
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
    let mut block = BlockHeader {
        version: 0x01000000u32.swap_bytes() as i32,
        prev_blockhash: BlockHash::from_hex(
            &"81cd02ab7e569e8bcd9317e2fe99f2de44d49ab2b8851ba4a308000000000000"
                .chars()
                .tuples()
                .map(|(a, b)| format!("{a}{b}"))
                .collect::<Vec<_>>()
                .into_iter()
                .rev()
                .collect::<String>(),
        )
        .unwrap(),
        merkle_root: TxMerkleNode::from_hex(
            &"e320b6c2fffc8d750423db8b1eb942ae710e951ed797f7affc8892b0f1fc122b"
                .chars()
                .tuples()
                .map(|(a, b)| format!("{a}{b}"))
                .collect::<Vec<_>>()
                .into_iter()
                .rev()
                .collect::<String>(),
        )
        .unwrap(),
        time: UNIX_EPOCH.elapsed().unwrap().as_secs() as u32,
        bits: BlockHeader::compact_target_from_u256(&Uint256::from_be_bytes(hex!(
            "0000000000000000000727200000000000000000000000000000000000000000"
        ))),
        nonce: 0,
    };
    eprintln!(
        "Estimated time: {:?}",
        Duration::from_secs(
            (block.work() / Uint256::from_u64(avg_hashes_per_second).unwrap()).low_u64()
        )
    );
    let block_slice = block_to_u32_slice(&block);
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
    let mut worker_slice = cuda.take_async(worker.clone()).unwrap();
    let initial = cuda.take_async(initial).unwrap();
    let target = cuda.take_async(
        block
            .target()
            .to_be_bytes()
            .into_iter()
            .rev()
            .tuples()
            .map(|(a, b, c, d)| u32::from_le_bytes([a, b, c, d]))
            .collect(),
    )?;
    let mut finished = cuda.take_async(vec![u32::MIN]).unwrap();
    let start = Instant::now();
    sha256.launch_async(
        LaunchConfig {
            block_dim: (block_size, 1, 1),
            grid_dim: (grid_size, 1, 1),
            shared_mem_bytes: 0,
        },
        (
            &hash_io,
            &mut worker_slice,
            &target,
            &initial,
            &mut finished,
        ),
    )?;
    cuda.synchronize()?;
    let elapsed = start.elapsed().as_secs_f64();
    let valid_header = if cuda.sync_release(finished)?[0] > u32::MIN {
        let output = cuda.sync_release(worker_slice)?;
        block.nonce = output[0].swap_bytes();
        assert!(block.validate_pow(&block.target()).is_ok());
        Some(block)
    } else {
        None
    };
    Ok((elapsed, valid_header))
}

fn main() {
    // about 10-100x faster than the hashrate stated online
    //let sizes = [4u32, 8, 16, 32, 64, 128, 256, 512, 1024];
    //let grid_block_size = sizes.into_iter().cartesian_product(sizes).collect_vec();
    let mut avg_hashes_per_second = (u32::MAX as f64) / 33.8;
    let mut i = 0;
    unsafe {
        loop {
        //for (grid_size, block_size) in grid_block_size.into_iter().skip(8) {
            //println!("\x1b[1m{grid_size:>4} | {block_size:>4}\x1b[0m");
            let grid_size = 32;
            let block_size = 64;
            match _main(block_size, grid_size, avg_hashes_per_second as u64) {
                Ok((elapsed, Some(block_header))) => {
                    eprintln!("Elapsed: {elapsed}s, nonce: {:#x}", block_header.nonce);
                    eprintln!("Hash: {}", block_header.block_hash());
                    break;
                }
                Ok((elapsed, ..)) => {
                    let new_hashes_per_second = (u32::MAX as f64) / elapsed;
                    avg_hashes_per_second =
                        i as f64 * avg_hashes_per_second + new_hashes_per_second;
                    i += 1;
                    avg_hashes_per_second /= i as f64;
                    eprintln!("Nothing found after {elapsed}s");
                    println!(
                        "BlockHash speed: {:.1}MH/s (avg: {:.1}MH/s)",
                        new_hashes_per_second / 1_000_000.0,
                        avg_hashes_per_second / 1_000_000.0
                    );
                }
                Err(e) => println!("Error {e}"),
            }
        }
    }
}

// Results: [grids, blocks, MH/s]
// [
//     [4, 8, 3],
//     [4, 16, 7],
//     [4, 32, 14],
//     [4, 64, 28],
//     [4, 128, 57],
//     [4, 512, 135],
//     [8, 4, 1],
//     [8, 8, 6],
//     [8, 16, 13],
//     [8, 32, 27],
//     [8, 64, 54],
//     [8, 128, 109],
//     [8, 256, 151],
//     [8, 512, 119],
//     [16, 4, 7],
//     [16, 8, 14],
//     [16, 16, 28],
//     [16, 32, 54],
//     [16, 64, 110],
//     [16, 128, 159],
//     [16, 256, 131],
//     [16, 512, 137],
//     [32, 4, 14],
//     [32, 8, 28],
//     [32, 16, 55],
//     [32, 32, 109],
//     [32, 64, 161],
//     [32, 128, 131],
//     [32, 256, 130],
//     [32, 512, 132],
//     [64, 4, 19],
//     [64, 8, 46],
//     [64, 16, 89],
//     [64, 32, 159],
//     [64, 64, 131],
//     [64, 128, 125],
//     [64, 256, 125],
//     [64, 512, 137],
//     [128, 4, 27],
//     [128, 8, 58],
//     [128, 16, 106],
//     [128, 32, 131],
//     [128, 64, 124],
//     [128, 128, 113],
//     [128, 256, 121],
//     [128, 512, 135],
//     [256, 4, 5],
//     [256, 8, 55],
//     [256, 16, 96],
//     [256, 32, 119],
//     [256, 64, 108],
//     [256, 128, 111],
//     [256, 256, 115],
//     [256, 512, 127],
//     [512, 4, 28],
//     [512, 8, 59],
//     [512, 16, 96],
//     [512, 32, 107],
//     [512, 64, 111],
//     [512, 128, 113],
//     [512, 256, 119],
//     [512, 512, 131],
//     [1024, 4, 30],
//     [1024, 8, 63],
//     [1024, 16, 103],
//     [1024, 32, 112],
//     [1024, 64, 52],
//     [1024, 128, 44],
//     [1024, 256, 54],
//     [1024, 512, 76],
// ];
