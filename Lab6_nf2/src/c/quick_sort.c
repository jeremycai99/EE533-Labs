//Sample quick sort implementation in C
int partition(int array[], int low, int high) {
    int pivot = array[high];
    int i, j, swap;

    i = low - 1;

    for (j = low; j < high; j++) {
        if (array[j] < pivot) {
            i++;

            swap = array[i];
            array[i] = array[j];
            array[j] = swap;
        }
    }

    swap = array[i + 1];
    array[i + 1] = array[high];
    array[high] = swap;

    return i + 1;
}

void quick_sort(int array[], int low, int high) {
    int p;

    if (low < high) {
        p = partition(array, low, high);

        quick_sort(array, low, p - 1);
        quick_sort(array, p + 1, high);
    }
}

int main() {
    int array[10] = {323, 123, -455, 2, 98, 125, 10, 65, -56, 0};
    int i;

    quick_sort(array, 0, 9);
    
    return 0;
}