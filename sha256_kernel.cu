typedef unsigned char uint8_t;
typedef unsigned int uint32_t;

__constant__ uint32_t K[64] = {
    0x428a2f98,
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
    0xc67178f2};

#define ROTATE_RIGHT(a, rotate) ((a >> rotate) | (a << (sizeof(a) * 8 - rotate)))
#define COMPRESSION(buffer, output, initial, a, b, c, d, e, f, g, h)                                                                                                            \
    do                                                                                                                                                                          \
    {                                                                                                                                                                           \
        a = initial[0];                                                                                                                                                         \
        b = initial[1];                                                                                                                                                         \
        c = initial[2];                                                                                                                                                         \
        d = initial[3];                                                                                                                                                         \
        e = initial[4];                                                                                                                                                         \
        f = initial[5];                                                                                                                                                         \
        g = initial[6];                                                                                                                                                         \
        h = initial[7];                                                                                                                                                         \
        for (int j = 16; j < 64; j++)                                                                                                                                           \
        {                                                                                                                                                                       \
            uint32_t s0 = buffer[j - 15], s1 = buffer[j - 2];                                                                                                                   \
            buffer[j] = buffer[j - 16] + buffer[j - 7] + (ROTATE_RIGHT(s0, 7) ^ ROTATE_RIGHT(s0, 18) ^ (s0 >> 3)) + (ROTATE_RIGHT(s1, 17) ^ ROTATE_RIGHT(s1, 19) ^ (s1 >> 10)); \
        }                                                                                                                                                                       \
                                                                                                                                                                                \
        for (int j = 0; j < 64; j++)                                                                                                                                            \
        {                                                                                                                                                                       \
            uint32_t t1 = h + (ROTATE_RIGHT(e, 6) ^ ROTATE_RIGHT(e, 11) ^ ROTATE_RIGHT(e, 25)) + ((e & f) ^ ((~e) & g)) + K[j] + buffer[j];                                     \
            uint32_t t2 = (ROTATE_RIGHT(a, 2) ^ ROTATE_RIGHT(a, 13) ^ ROTATE_RIGHT(a, 22)) + ((a & b) ^ (a & c) ^ (b & c));                                                     \
                                                                                                                                                                                \
            h = g;                                                                                                                                                              \
            g = f;                                                                                                                                                              \
            f = e;                                                                                                                                                              \
            e = d + t1;                                                                                                                                                         \
            d = c;                                                                                                                                                              \
            c = b;                                                                                                                                                              \
            b = a;                                                                                                                                                              \
            a = t1 + t2;                                                                                                                                                        \
        }                                                                                                                                                                       \
        output[0] = initial[0] + a;                                                                                                                                             \
        output[1] = initial[1] + b;                                                                                                                                             \
        output[2] = initial[2] + c;                                                                                                                                             \
        output[3] = initial[3] + d;                                                                                                                                             \
        output[4] = initial[4] + e;                                                                                                                                             \
        output[5] = initial[5] + f;                                                                                                                                             \
        output[6] = initial[6] + g;                                                                                                                                             \
        output[7] = initial[7] + h;                                                                                                                                             \
    } while (0)

__device__ void _sha256(uint32_t input[64], uint32_t output[64], const uint32_t initial[8], const uint32_t worker[8])
{
    uint32_t a, b, c, d, e, f, g, h;
    COMPRESSION(input, output, worker, a, b, c, d, e, f, g, h);
    COMPRESSION(output, output, initial, a, b, c, d, e, f, g, h);
}

extern "C" __global__ void sha256(const uint32_t io[16], uint32_t w[8], uint8_t target[32], const uint32_t i[8], uint32_t finished[1])
{
    uint32_t initial = blockIdx.x * blockDim.x + threadIdx.x, nonce = initial;
    uint32_t step = gridDim.x * blockDim.x;

    uint32_t _in[64];
    memcpy(_in, io, 16 * 4);
    uint32_t _out[64];
    uint8_t *_bout = (uint8_t *)_out;
    _out[8] = 1 << 31;
    memset(_out + 9, 0, (15 - 9) * 4);
    _out[15] = 256;
    while (finished[0] == 0)
    {
        _in[3] = nonce;
        _sha256(_in, _out, i, w);
        for (int b = 31; b >= 0; b--)
        {
            if (_bout[b] < target[b] && false)
            {
                if (atomicAdd(finished, 1) == 0)
                {
                    memcpy(w, _out, 7 * 4);
                    w[7] = nonce;
                }
                return;
            }
            else if (_bout[b] > target[b])
                break;
        }
        nonce += step;
        // this means an overflow occurred
        if (nonce < step)
            return;
    }
}