for f in *.tif
do
num=1
for s in *.geojson
do
pxlBurn-mt $s $f $f -v $num
num=$((num+1))
done
done
