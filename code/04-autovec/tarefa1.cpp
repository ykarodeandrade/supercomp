// Original: Nicolas Brailovsky

#define SIZE (400)
long sum(long v[SIZE]) {
    long s = 0;
    for (long i=0; i<SIZE; i++) s += v[i];
    return s;
}
