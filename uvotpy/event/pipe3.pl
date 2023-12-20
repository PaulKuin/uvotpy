#! /usr/bin/perl

# Quick and dirty pipeline to go from a Swift/UVOT RAW image to a SKY image

# Stephen Holland, 2005 Feb 16, v0.1 (first try)
# Stephen Holland, 2005 Mar  1, v0.2 (streamline it a bit)
# Stephen Holland, 2005 Mar 17, v0.3 (reads RA,Dec from files)
# Paul Kuin 2008 Sep 9; simplify (no bad pix or mod8 - for getting just a good sky header 

use strict;
use warnings;

use Astro::FITS::CFITSIO qw(:constants :longnames);

my $chatter = 1;
my $cleanup = "YES";
my $clobber = "YES";
my $history = "YES";

# Calibration files
my $badpixlist = "CALDB";
my $flatfile   = "CALDB";
my $alignfile  = "CALDB";
my $teldef     = "CALDB";
my $zerofile   = "CALDB";
my $coinfile   = "CALDB";

my $command = "";
my $comment = "";
my $status = 0;

die "usage: $0 infile_root [attitude_file]\n" if ( $#ARGV < 0 );

my $file_root = $ARGV[0];

# Set the name of the attitude file
my $attitude_file = "";
if ( $#ARGV == 0 ) {
    my $root = $1 if ( $file_root =~ m/(sw\d{11})u\w{2}/ );
    $attitude_file = "../../auxil/".$root."sat.fits";
} else {
    $attitude_file = $ARGV[1];
}

# Check that the attitude file exists.
die "error, $attitude_file does not exist, $!\n" unless ( -e $attitude_file );

# Extract the RA and Dec from the RAW image.
my $ra = -99.999;
my $dec = -99.999;
my $extension = $file_root."_rw.img+1";
my $fptr = undef;
&fits_open_file($fptr,$extension,READONLY(),$status);
die "error opening $extension for reading, status = $status, $!\n" if ( $status != 0 );
&fits_read_keyword($fptr,'RA_PNT',$ra,$comment,$status);
warn "warning, unable to read RA_PNT from $extension, status = $status, $!\n" if ( $status != 0 );
&fits_read_keyword($fptr,'DEC_PNT',$dec,$comment,$status);
warn "warning, unable to read DEC_PNT from $extension, status = $status, $!\n" if ( $status != 0 );
&fits_close_file($fptr,$status);
warn "warning unable to close $extension, status = $status, $!\n" if ( $status != 0 );

my $raw_file = $file_root."_rw.img";
die "error, $raw_file does not exist, $!\n" unless ( -e $raw_file );

# Print some header information.
print "\n";
print "      Command: $0 @ARGV\n";
my $date = `date`;
chomp($date);
print "         Date: $date\n";
print "\n";
print "     RAW data: $raw_file\n";
print "Attitude data: $attitude_file\n";
print "     Pointing: $ra, $dec\n";

## Make the bad pixel list.
#print "\n";
#print "Make the bad pixel list...\n";
#my $bp_file = $file_root."_bp.img";
#if ( $clobber ) {
#    warn "warning, existing $bp_file will be overwritten" if ( -e $bp_file );
#} else {
#    die "error, $bp_file already exists" if ( -e $bp_file );
#}
#$command = "uvotbadpix infile=$raw_file badpixlist=$badpixlist outfile=$bp_file compress=YES clobber=$clobber history=$history chatter=$chatter";
##print "$command\n";
#system( $command ) && die "system call failed for \"$command\", $!";

# Make the mod-8 map.
#print "\n";
#print "Make the mod-8 map...\n";
#my $md_file = $file_root."_md.img";
#if ( $clobber ) {
#    warn "warning, existing $md_file will be overwritten" if ( -e $md_file );
#} else {
#    die "error, $md_file already exists" if ( -e $md_file );
#}
#my $modmap = $file_root."_mm.img";
#if ( $clobber ) {
#    warn "warning, existing $modmap will be overwritten" if ( -e $modmap );
#} else {
#    die "error, $modmap already exists" if ( -e $modmap );
#}
#$command = "uvotmodmap infile=$raw_file badpixfile=$bp_file outfile=$md_file mod8prod=NO mod8file=$modmap nsig=3 ncell=16 subimage=NO xmin=0 xmax=2047 ymin=0 ymax=2047 clobber=$clobber history=$history chatter=$chatter";
#print "$command\n";
#system( $command ) && die "system call failed for \"$command\", $!";

#my $ff_file = $md_file;
my $ff_file = $raw_file;

# Transform to SKY coordinates
print "\n";
print "Transform to SKY coordinates...\n";
my $sk_file = $file_root."_sk.img";
if ( $clobber ) {
    warn "warning, existing $sk_file will be overwritten" if ( -e $sk_file );
} else {
    die "error, $sk_file already exists" if ( -e $sk_file );
}
die "error, $attitude_file does not exist, $!" unless ( -e $attitude_file );
$command = "swiftxform infile=$ff_file outfile=$sk_file attfile=$attitude_file alignfile=$alignfile method=AREA to=SKY ra=$ra dec=$dec roll=0.0 teldeffile=$teldef bitpix=-32 zeronulls=NO aberration=NO seed=-1956 copyall=NO extempty=YES allempty=NO history=$history clobber=$clobber cleanup=$cleanup chatter=$chatter";
#print "$command\n";
system( $command ) && die "system call failed for \"$command\", $!";

