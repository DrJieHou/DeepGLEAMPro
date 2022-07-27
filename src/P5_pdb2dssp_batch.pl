#!/usr/bin/perl

#######pdb2dssp.pl############################## 
#convert pdb format file to dssp format file.
#usage: pdb2dssp.pl source_dir dest_dir
#the dssp file will have the same file name with pdb file except with differnt
#suffix ".dssp.gz". The dssp file is compressed by gzip. 
#Assumption: source file name format: *.Z and compress by gzip. 
#output: pdb_prefix.dssp.gz 
#Author: Jianlin Cheng, 5/28/2003
###############################################

if (@ARGV != 3)
{
  die "Need three arguments: dssp_dir, source_dir dest_dir\n"
}

$dssp_dir = shift @ARGV; 
$source_dir =  shift @ARGV;
$dest_dir =  shift @ARGV;


`mkdir -p $dest_dir`;
opendir(SOURCE_DIR, "$source_dir") || die "can't open the source dir!";
if (! -d "$dest_dir")
{
  die "can't open the dest dir!";
}


@filelist = readdir(SOURCE_DIR);

while(@filelist) 
{ 
   $pdbfile = shift @filelist;
   $full_pdb_name = $source_dir.'/'.$pdbfile;
   #if ( -f $full_pdb_name && $full_pdb_name =~ /.*Z$/)
   if ( -f $full_pdb_name && (index($full_pdb_name,'TS')>0 or index($full_pdb_name,'.pdb')>0))
   {
       #print "$full_pdb_name\n";

       #`gzip -f -d $temp_file`;       
       $unzip_file = $full_pdb_name;
       $dssp_file = $dest_dir."/$pdbfile.dssp";
       #do conversion
       #print "${dssp_dir}dsspcmbi $unzip_file $dssp_file\n";
       $status = system("${dssp_dir}/dsspcmbi $unzip_file $dssp_file");
       if ($status == 0) #succeed
       {
          #`gzip -f $dssp_file`;
       }
       else
       {
          `rm $dssp_file`;
       }
       #remove the unzipped pdb file
   }
}

closedir(SOURCE_DIR);



