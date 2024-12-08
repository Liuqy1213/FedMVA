#!/bin/bash
VERBOSE=0

cat < /dev/null > BadFileList
#  ./slicer_all.sh  test 2 test-output

if [[ $# -ge 1 ]]; then
    inpdir=$1
    outdir=$3;
    dfonly='';

    mkdir -p 'tmp';
    mkdir -p 'ParseOutput';

    #cp $inpdir'/'$filename 'tmp/'$filename;


#find test -name '*.c' -exec bash -c '
find $inpdir -name '*.c' -exec bash -c '
  # For every C files found
  lineno=22
  outdir="out-dir"
  for item do
    #Get the filename
    Out=$( echo $item  | sed "s/.*\///" )

    echo $Out

## call Joern
    rm -rf tmp/*
    rm -rf out-dir/*
    cp $item tmp
joern/joern-parse tmp/ $Out

#mkdir 'ParseOutput/'$Out

# Copy the Joern nodes and edges into a directory
# copy from the instance temporary into a named output

cp -r './tmp/'$Out   'ParseOutput/'$Out

   if [[ $VERBOSE = 1 ]]; then
      VERB='--verbose';
   fi

echo $Out
#echo $lineno
#echo $lineno
#echo $outdir

PyOut='ParseOutput/'$Out

# Should Re-write this to output to the correct directory
# python parse_joern_output_quiet.py --code $Out --line $lineno --output $outdir $dfonly $VERB;

Ofile="out-dir/"$Out".forward"
echo $Ofile
#if [[ -f $Ofile ]]
#then
#	cp  'out-dir/'$Out'.forward'   'ParseOutput/'$Out
#	cp  'out-dir/'$Out'.backward'   'ParseOutput/'$Out
#else
#	echo "Python Joern Parse Failed"
#	echo $Out >> BadFileList
#fi
done

' bash {} \;

else
  echo 'Wrong Argument!.'
  echo 'slicer_all.sh <Directory of the C File>  Line number of two is assumed , output is hard coded' 
fi



