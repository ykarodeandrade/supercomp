#define SIZE (5)
long sum(int v[SIZE]) {
    long s = 0;
    for (unsigned i=0; i<SIZE; i++) s += v[i];
    return s;
    }