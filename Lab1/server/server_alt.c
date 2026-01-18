/* Alternative method for server code design */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/types.h> 
#include <sys/socket.h>
#include <netinet/in.h>

void error(char *msg)
{
    perror(msg);
    exit(1);
}

void dostuff(int sock)
{
    char buffer[256];
    int n;
    struct sockaddr_in cli_addr;
    socklen_t clilen;

    while (1) {
        bzero((struct sockaddr *)&cli_addr, sizeof(cli_addr));
        clilen = sizeof(cli_addr);

        bzero(buffer, 256);
        n = recvfrom(sock, buffer, 255, 0,
                     (struct sockaddr *)&cli_addr, &clilen);
        if (n < 0) error("ERROR reading from socket"); // Changed to more user friendly message

        printf("Here is the message: %s\n", buffer);

        n = sendto(sock, "I got your message", 18, 0,
                   (struct sockaddr *)&cli_addr, clilen);
        if (n < 0) error("ERROR writing to socket"); // Changed to more user friendly message
    }
}

int main(int argc, char *argv[])
{
     int sock, portno;
     struct sockaddr_in serv_addr;
     if (argc < 2) {
         fprintf(stderr,"ERROR, no port provided\n");
         exit(1);
     }
     sock = socket(AF_INET, SOCK_DGRAM, 0);
     if (sock < 0) 
        error("ERROR opening socket");
     bzero((char *) &serv_addr, sizeof(serv_addr));
     portno = atoi(argv[1]);
     serv_addr.sin_family = AF_INET;
     serv_addr.sin_addr.s_addr = INADDR_ANY;
     serv_addr.sin_port = htons(portno);
     if (bind(sock, (struct sockaddr *) &serv_addr,
              sizeof(serv_addr)) < 0) 
              error("ERROR on binding");

     dostuff(sock);

     close(sock);
     return 0; 
}
