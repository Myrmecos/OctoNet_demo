# unzip all zip files
find node_*/ -name "*.zip" -exec sh -c '
  for zip; do
    dir="${zip%.zip}"
    mkdir -p "$dir"
    unzip -j -o "$zip" -d "$dir"
  done
' _ {} +

# remove all zip files
find node_*/ -name "*.zip" -exec sh -c '
  for zip; do
    rm -f "$zip"
  done
' _ {} +


# move all zip files to unused folder
for folder in node_* meta; do
  for file in "$folder"/*.zip; do
    mv "$file" unused/
  done
done

