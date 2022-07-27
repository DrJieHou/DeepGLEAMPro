#!/usr/bin/perl -w

if (@ARGV != 3 ) {
  print "Usage: <input> <output>\n";
  exit;
}

$dnss_file = $ARGV[0];
$targetid = $ARGV[1];
$dnss_out = $ARGV[2];


if(!(-e $dnss_file))
{
   die "Failed to find $dnss_file\n";
}
  open(FILE,$dnss_file) || die "Failed to open file $dnss_file\n";
  @ss_contents = <FILE>;
  close FILE;
  shift @ss_contents;
  shift @ss_contents;
  $dnss_ss = shift @ss_contents;
  chomp $dnss_ss;




  open(TMPOUT,">$dnss_out") || die "Failed to open file $dnss_out\n";
  print TMPOUT ">$targetid\n".$dnss_ss."\n";
  close TMPOUT;

  
