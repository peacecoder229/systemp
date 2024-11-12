#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <inttypes.h>





static void
print_stats(void)
{
    uint64_t total_packets_dropped, total_packets_tx, total_packets_rx;
    unsigned portid;
    total_packets_dropped = 0;
    total_packets_tx = 0;
    total_packets_rx = 0;
    const char clr[] = { 27, '[', '2', 'J', '\0' };
    const char topLeft[] = { 27, '[', '1', ';', '1', 'H','\0' };
        /* Clear screen and move to top left */
    printf("%s%s", clr, topLeft);
    printf("\nPort statistics ====================================");

}




int main(void) {
	print_stats();

printf("\n====================================================\n");
    fflush(stdout);



}







