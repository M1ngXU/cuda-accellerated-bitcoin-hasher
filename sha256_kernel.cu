#define ROTATE_RIGHT(a, rotate) ((a >> rotate) | (a << (sizeof(a) * 8 - rotate)))
typedef unsigned int uint32_t;

extern "C" __global__ void sha256(uint32_t i[64], uint32_t w[8], const uint32_t k[64])
{
    uint32_t h0 = w[0], a = h0,
             h1 = w[1], b = h1,
             h2 = w[2], c = h2,
             h3 = w[3], d = h3,
             h4 = w[4], e = h4,
             h5 = w[5], f = h5,
             h6 = w[6], g = h6,
             h7 = w[7], h = h7;
    for (int j = 16; j < 64; j++)
    {
        uint32_t s0 = i[j - 15], s1 = i[j - 2];
        i[j] = i[j - 16] + i[j - 7] + (ROTATE_RIGHT(s0, 7) ^ ROTATE_RIGHT(s0, 18) ^ (s0 >> 3)) + (ROTATE_RIGHT(s1, 17) ^ ROTATE_RIGHT(s1, 19) ^ (s1 >> 10));
    }

    for (int j = 0; j < 64; j++)
    {
        uint32_t t1 = h + (ROTATE_RIGHT(e, 6) ^ ROTATE_RIGHT(e, 11) ^ ROTATE_RIGHT(e, 25)) + ((e & f) ^ ((~e) & g)) + k[j] + i[j];
        uint32_t t2 = (ROTATE_RIGHT(a, 2) ^ ROTATE_RIGHT(a, 13) ^ ROTATE_RIGHT(a, 22)) + ((a & b) ^ (a & c) ^ (b & c));

        h = g;
        g = f;
        f = e;
        e = d + t1;
        d = c;
        c = b;
        b = a;
        a = t1 + t2;
    }
    i[0] = h0 + a;
    i[1] = h1 + b;
    i[2] = h2 + c;
    i[3] = h3 + d;
    i[4] = h4 + e;
    i[5] = h5 + f;
    i[6] = h6 + g;
    i[7] = h7 + h;
}