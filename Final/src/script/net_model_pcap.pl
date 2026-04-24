#!/usr/bin/perl
#
# net_model_pcap.pl
#
# Convert model_test-emitted FRAME lines into Ethernet pcap files for replay
# with tcpreplay on the NetFPGA host.
#
# This script only wraps Final/src/script/model_test emit_* output.  It does
# not generate any old Lab6 test payloads.
#
# Example:
#   MODEL_DIR=Final/model SOC_EMIT_PATH=/tmp/ann_classes.frames \
#     perl Final/src/script/model_test emit_classes -q
#
#   perl Final/src/script/net_model_pcap.pl \
#     --in /tmp/ann_classes.frames \
#     --out /tmp/ann_classes.pcap \
#     --src-mac 00:4e:46:32:43:00 \
#     --dst-mac ff:ff:ff:ff:ff:ff

use strict;

my $in_path = "";
my $out_path = "socnet_frames.pcap";
my $src_mac = "00:4e:46:32:43:00";
my $dst_mac = "ff:ff:ff:ff:ff:ff";
my $ethertype = 0x88B5;
my $gap_us = 20000;
my $cpu_gap_us = 250000;

sub usage {
    print "Usage: perl net_model_pcap.pl --in frames.txt --out out.pcap [options]\n";
    print "Options:\n";
    print "  --src-mac MAC       default 00:4e:46:32:43:00\n";
    print "  --dst-mac MAC       default ff:ff:ff:ff:ff:ff\n";
    print "  --ethertype HEX     default 0x88B5\n";
    print "  --gap-us N          timestamp gap after normal frames, default 20000\n";
    print "  --cpu-gap-us N      timestamp gap after cpu frames, default 250000\n";
}

sub parse_args {
    while (@ARGV) {
        my $a = shift @ARGV;
        if ($a eq "--in")        { $in_path = shift @ARGV; }
        elsif ($a eq "--out")    { $out_path = shift @ARGV; }
        elsif ($a eq "--src-mac") { $src_mac = shift @ARGV; }
        elsif ($a eq "--dst-mac") { $dst_mac = shift @ARGV; }
        elsif ($a eq "--ethertype") { $ethertype = hex_or_dec(shift @ARGV); }
        elsif ($a eq "--gap-us") { $gap_us = int(shift @ARGV); }
        elsif ($a eq "--cpu-gap-us") { $cpu_gap_us = int(shift @ARGV); }
        elsif ($a eq "--help" || $a eq "-h") { usage(); exit 0; }
        else { die "Unknown option '$a'\n"; }
    }
    if ($in_path eq "") {
        usage();
        die "Missing --in\n";
    }
}

sub hex_or_dec {
    my ($s) = @_;
    return hex($1) if $s =~ /^0x([0-9a-fA-F]+)$/;
    return int($s);
}

sub mac_bytes {
    my ($s) = @_;
    my @p = split(/:/, $s);
    die "Bad MAC '$s'\n" unless @p == 6;
    my $b = "";
    for my $x (@p) {
        die "Bad MAC byte '$x' in '$s'\n" unless $x =~ /^[0-9a-fA-F]{1,2}$/;
        $b .= pack("C", hex($x) & 0xFF);
    }
    return $b;
}

sub u64_bytes {
    my ($s) = @_;
    $s =~ s/^0x//i;
    die "Bad 64-bit word '$s'\n" unless $s =~ /^[0-9a-fA-F]{1,16}$/;
    $s = ("0" x (16 - length($s))) . $s;
    my $hi = hex(substr($s, 0, 8));
    my $lo = hex(substr($s, 8, 8));
    return pack("NN", $hi & 0xFFFFFFFF, $lo & 0xFFFFFFFF);
}

sub write_pcap_global_header {
    my ($fh) = @_;
    print $fh pack("VvvVVVV", 0xa1b2c3d4, 2, 4, 0, 0, 65535, 1);
}

sub write_pcap_packet {
    my ($fh, $frame, $ts_us) = @_;
    $frame .= "\x00" x (64 - length($frame)) if length($frame) < 64;
    my $sec = int($ts_us / 1000000);
    my $usec = $ts_us % 1000000;
    my $len = length($frame);
    print $fh pack("VVVV", $sec, $usec, $len, $len);
    print $fh $frame;
}

sub build_eth_frame {
    my ($words_ref) = @_;
    my $payload = "\x00\x00";  # Final pkt_proc network alignment pad.
    for my $w (@$words_ref) {
        $payload .= u64_bytes($w);
    }
    return mac_bytes($dst_mac) . mac_bytes($src_mac) . pack("n", $ethertype) . $payload;
}

sub parse_frame_line {
    my ($line, $lineno) = @_;
    my @tok = split(/\s+/, $line);
    die "Line $lineno: expected FRAME\n" unless @tok >= 3 && $tok[0] eq "FRAME";
    my $idx = int($tok[1]);
    my %meta;
    my @words;
    my $i = 2;
    while ($i < @tok && $tok[$i] =~ /=/) {
        my ($k, $v) = split(/=/, $tok[$i], 2);
        $meta{$k} = $v;
        $i++;
    }
    while ($i < @tok) {
        push @words, $tok[$i];
        $i++;
    }
    if (exists $meta{"words"} && int($meta{"words"}) != @words) {
        die "Line $lineno: metadata words=$meta{words}, got " . scalar(@words) . "\n";
    }
    return ($idx, \%meta, \@words);
}

parse_args();

open(my $in_fh, "<", $in_path) or die "Cannot read $in_path: $!\n";
open(my $out_fh, ">", $out_path) or die "Cannot write $out_path: $!\n";
binmode $out_fh;
write_pcap_global_header($out_fh);

my $lineno = 0;
my $frame_count = 0;
my $expect_count = 0;
my $ts_us = 0;

while (my $line = <$in_fh>) {
    $lineno++;
    chomp $line;
    $line =~ s/^\s+//;
    $line =~ s/\s+$//;
    next if $line eq "" || $line =~ /^#/;

    my ($idx, $meta_ref, $words_ref) = parse_frame_line($line, $lineno);
    my $frame = build_eth_frame($words_ref);
    write_pcap_packet($out_fh, $frame, $ts_us);
    $frame_count++;
    $expect_count++ if exists $meta_ref->{"expect"} && $meta_ref->{"expect"} eq "1";
    $ts_us += (exists $meta_ref->{"cpu"} && $meta_ref->{"cpu"} eq "1") ? $cpu_gap_us : $gap_us;
}

close $in_fh;
close $out_fh;

printf("Wrote %s: %d Ethernet frames (%d expect responses), ethertype=0x%04X\n",
       $out_path, $frame_count, $expect_count, $ethertype);
print "Replay example:\n";
print "  tcpreplay -i nf2c0 $out_path\n";
print "Capture example:\n";
print "  tcpdump -i nf2c1 -nn -e -xx ether proto 0x" . sprintf("%04x", $ethertype) . "\n";

