//Sample bubble sort implementation in C

int main() { int i, j, swap;
    int array[10] = {323, 123, -455, 2, 98, 125, 10, 65, -56, 0};
    for (i = 0 ; i < 10; i++) {
        for (j = i+1 ; j < 10 ; j++) {
            if (array[j] < array[i]) {
                swap = array[j];
                array[j] = array[i];
                array[i] = swap;
            }
        }
    }
}