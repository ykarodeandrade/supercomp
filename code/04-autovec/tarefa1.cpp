// Original: Nicolas Brailovsky

#define SIZE (400)
long sum(long v[SIZE]) {
    long *d = new long[SIZE];
    long s = 0; // este exemplo eh didatico. soma de ints deveria ser long ;)
    for (long i=0; i<SIZE; i++) s += v[i];
    return s;
}
