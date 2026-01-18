#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include <sys/types.h>
#include <sys/socket.h>
#include <netinet/in.h>

void error(const char *msg)
{
    perror(msg);
    exit(1);
}

int main(int argc, char *argv[])
{
    int sock;
    int portno;

    struct sockaddr_in serv_addr;
    struct sockaddr_in from;
    socklen_t fromlen;

    int n;
    char buf[1024];

    if (argc < 2) {
        fprintf(stderr, "ERROR, no port provided\n");
        exit(1);
    }

    sock = socket(AF_INET, SOCK_DGRAM, 0);
    if (sock < 0)
        error("opening socket");

    bzero((char *)&serv_addr, sizeof(serv_addr));
    portno = atoi(argv[1]);
    serv_addr.sin_family = AF_INET;
    serv_addr.sin_addr.s_addr = INADDR_ANY;
    serv_addr.sin_port = htons(portno);

    if (bind(sock, (struct sockaddr *)&serv_addr, sizeof(serv_addr)) < 0)
        error("binding");

    fromlen = sizeof(struct sockaddr_in);

    while (1) {
        bzero(buf, 1024);

        n = recvfrom(sock, buf, 1024, 0, (struct sockaddr *)&from, &fromlen);
        if (n < 0)
            error("recvfrom");

        printf("Here is the message: %s\n", buf);

        n = sendto(sock, "Got your message", 17, 0, (struct sockaddr *)&from, fromlen);
        if (n < 0)
            error("sendto");
    }

    return 0;
}