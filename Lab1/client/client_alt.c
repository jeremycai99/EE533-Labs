/* Alternative method for client code design */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include <sys/types.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <netdb.h>

static void error(const char *msg)
{
    perror(msg);
    exit(1);
}

int main(int argc, char *argv[])
{
    int sock, portno, n;
    socklen_t serverlen, fromlen;

    struct sockaddr_in serveraddr, from;
    struct hostent *server;

    char buf[1024];

    if (argc < 3) {
        fprintf(stderr, "usage %s hostname port\n", argv[0]);
        exit(0);
    }

    portno = atoi(argv[2]);

    sock = socket(AF_INET, SOCK_DGRAM, 0);
    if (sock < 0) error("opening socket");

    server = gethostbyname(argv[1]);
    if (server == NULL) {
        fprintf(stderr, "ERROR, no such host\n");
        exit(0);
    }

    bzero((char *)&serveraddr, sizeof(serveraddr));
    serveraddr.sin_family = AF_INET;
    bcopy((char *)server->h_addr, (char *)&serveraddr.sin_addr.s_addr, server->h_length);
    serveraddr.sin_port = htons(portno);

    serverlen = sizeof(serveraddr);

    printf("Please enter the message: ");
    bzero(buf, 1024);
    fgets(buf, 1023, stdin);

    n = sendto(sock, buf, strlen(buf), 0, (struct sockaddr *)&serveraddr, serverlen);
    if (n < 0) error("sendto");

    fromlen = sizeof(from);
    bzero(buf, 1024);
    n = recvfrom(sock, buf, 1024, 0, (struct sockaddr *)&from, &fromlen);
    if (n < 0) error("recvfrom");

    printf("Server reply: %s\n", buf);

    close(sock);
    return 0;
}