find . -type f -name "*parquet" | grep -v -e "37650" -e "90954" | grep -v "BAD" | while read file; do
  dir=$(dirname "$file")
  name=$(basename "$file")
  base="${name%.*}"
  ext="${name##*.}"
  if [ "$base" != "$name" ]; then
    mv "$file" "$dir/${base}_1y.$ext"
    # echo 'mv "$file" "$dir/${base}_1y.$ext"'
  else
    echo "$file $dir/${base}_1y"
    # echo 'mv "$file" "$dir/${base}_1y"'
  fi
done
