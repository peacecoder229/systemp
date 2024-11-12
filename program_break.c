#include <unistd.h>
#include <stdio.h>
#include <stdlib.h>

void program_break_test() {
  printf("%10p\n", sbrk(0));

  char *bl = malloc(1024 * 1024);
  printf("%10p\n", sbrk(0));
  printf("malloc'd at: %10p\n", bl);

  free(bl);
  printf("%10p\n", sbrk(0));

}

int main(int argc, char **argv) {
  program_break_test();
  return 0;
}
