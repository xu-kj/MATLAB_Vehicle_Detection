clear all;

% Generate a video clip using the processed pictures

WriterObj=VideoWriter('output_0704.avi');
% 'video_file.avi'表示待合成的视频（不仅限于avi格式）的文件路径
WriterObj.FrameRate = 15; % FrameRate选择帧率
WriterObj.Quality = 100;  % Quality选择视频质量

open(WriterObj);
img_file_dir = 'output_0704\'; % img_file_dir选择单帧图片存放路径
Readlist.Img_file = dir(fullfile(img_file_dir,'*.jpg'));
for i = 1:length(Readlist.Img_file)
    img_file = [img_file_dir, Readlist.Img_file(i).name];
    im = imread(img_file);
    writeVideo(WriterObj, im);
end
close(WriterObj);