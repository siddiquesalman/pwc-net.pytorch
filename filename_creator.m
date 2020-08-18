clc
clearvars
close all
fid_r = fopen('FlyingChairs_train_val.txt', 'r');

split = textscan(fid_r, '%s');
split = split{1};
tc = 1;
vc = 1;
foldr = 'LenslessDesign/datasets/FlyingChairs_release/data/';
fnames = dir(['/media/data/salman/',foldr,'*.flo']);
for i=1:length(split)
    file_name = fnames(i).name(1:end-9);
    full_file = [foldr,file_name];
    if str2num(split{i,1}) == 1
        files1{tc,1} = full_file;
        tc = tc+1;
    else
        files2{vc,1} = full_file;
        vc = vc+1;
    end
end
fid_w1 = fopen('train_flyingchairs.txt', 'w');
fid_w2 = fopen('val_flyingchairs.txt', 'w');
fprintf(fid_w1, '%s\n', files1{:});
fprintf(fid_w2, '%s\n', files2{:});
fclose(fid_w1);
fclose(fid_w2);    
    
