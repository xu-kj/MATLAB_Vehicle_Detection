%% Database Processing
function training()
    fprintf('Please select training data set: \n');
    x = input('Select training data set => ','s');
    if strcmp(x, 'UIUC') == 1
        model.svm = UIUC_train();
        save svmModel model
    elseif strcmp(x, 'IMAGENET') == 1
        model.svm = IMAGENET_train();
        save svmModel model
    elseif strcmp(x, 'CASE') == 1
        model.svm = CASE_train();
        save svmModel model
    else
        fprintf('Invalid input! \n');
    end
end

%% UIUC database processing
function svm = UIUC_train()
    img_file_dir = 'pictures\UIUC\101_ObjectCategories\car_side\';
    ann_file_dir = 'pictures\UIUC\Annotations\car_side\';
    Readlist.Img_file = dir(fullfile(img_file_dir,'*.jpg'));
    Readlist.Ann_file = dir(fullfile(ann_file_dir,'*.mat'));
    model.label = [];
    model.insc = [];

    for i = 1:length(Readlist.Img_file)
        img_file = [img_file_dir, Readlist.Img_file(i).name];
        ann_file = [ann_file_dir, Readlist.Ann_file(i).name];
        % ‘UIUC’数据库需要另外通过annotation.m脚本读出边框坐标
        contour = show_annotation(img_file, ann_file);       
        % 对每一张图片，转换成单字节格式黑白图片的格式
        im = imread(img_file);
        try
            img_single = im2single(rgb2gray(im));
        catch
            img_single = im2single(im);
        end
        img_car = img_single(contour(1):contour(2), contour(3):contour(4));
        img_single(contour(1):contour(2), contour(3):contour(4)) = 0;
        % 将正负样本都归一化至标准大小，这里选择64x64
        car.img = imresize(img_car, [64, 64]);
        bac.img = imresize(img_single, [64, 64]);
        % 通过vl_hog()函数计算出梯度分布直方图
        % 这里输入变量8表示已8x8为单元格计算
        car.hog = vl_hog(car.img, 8);
        bac.hog = vl_hog(bac.img, 8);
        hog_sz = size(car.hog);
        
        % 通过svm_hog()函数整理出正、负样本的HOG表达式，适合svmtrain()处理
        % car sample input
        [model.label, model.insc] = svm_hog(car.hog, hog_sz, model.label, model.insc, 1);
        % background sample input
        [model.label, model.insc] = svm_hog(bac.hog, hog_sz, model.label, model.insc, -1);
    end
    
    % svmtrain()要求输入为double类型，所以通过double()函数进行格式转换
    model.label = double(model.label);
    model.insc = double(model.insc);
    % 通过已经得到的包含了正、负样本的HOG数据矩阵，用svmtrain()计算样本svm值
    model.svm = svmtrain(model.label, model.insc, '-s 0 -t 0');
    svm = model.svm;
end 

%% IMAGENET database processing
function svm = IMAGENET_train()
    %training with IMAGENET database and output the parameters
    % van_file_dir = 'pictures\IMAGENET\van\';
    vehicle_file_dir = 'pictures\IMAGENET\vehicle_0522\';
    negative_file_dir = 'pictures\NEGATIVE\0524\';
    % Readlist.Van_file = dir(fullfile(van_file_dir,'*.JPEG'));
    Readlist.Vehicle_file = dir(fullfile(vehicle_file_dir,'*.JPEG'));
    Readlist.Negative_file = dir(fullfile(negative_file_dir,'*.jpg'));

    model.label = [];
    model.insc = [];

    % 'IMAGENET'数据库，先整理添加正样本
    for i = 1:length(Readlist.Vehicle_file)
        img_file = [vehicle_file_dir, Readlist.Vehicle_file(i).name];
        
        im = imread(img_file);
        try
            img_single = im2single(rgb2gray(im));
        catch
            img_single = im2single(im);
        end
        img_car = img_single;
        car.img = imresize(img_car, [64, 64]);
        
        car.hog = vl_hog(car.img, 8);
        hog_sz = size(car.hog);
%         sample_num = hog_sz(1) * hog_sz(2);
        
        %% car sample input
        [model.label, model.insc] = svm_hog(car.hog, hog_sz, model.label, model.insc, 1);
    end
    
    % for i = 1:length(Readlist.Van_file)
    %     img_file = [van_file_dir, Readlist.Van_file(i).name];
    %     
    %     im = imread(img_file);
    %     try
    %         img_single = im2single(rgb2gray(im));
    %     catch
    %         img_single = im2single(im);
    %     end
    %     img_car = img_single;
    %     car.img = imresize(img_car, [64, 64]);
    %     
    %     car.hog = vl_hog(car.img, 8);
    %     hog_sz = size(car.hog);
    %     sample_num = hog_sz(1) * hog_sz(2);
    %     
    %     %% car sample input
    %     model.label = [model.label; 1];
    %     car.insc = [];
    %     for sp_num = 1:sample_num
    %         for hog_num = 1:hog_sz(3)
    %             col = ceil(sp_num / hog_sz(1));
    %             row = sp_num - (col - 1) * hog_sz(1);
    %             car.insc = [car.insc, car.hog(row, col, hog_num)];
    %         end
    %     end
    %     model.insc = [model.insc; car.insc];
    % end

    % 再整理添加负样本
    for i = 1:length(Readlist.Negative_file)
        img_file = [negative_file_dir, Readlist.Negative_file(i).name];
        
        im = imread(img_file);
        try
            img_single = im2single(rgb2gray(im));
        catch
            img_single = im2single(im);
        end
        img_car = img_single;
        car.img = imresize(img_car, [64, 64]);
        
        car.hog = vl_hog(car.img, 8);
        hog_sz = size(car.hog);
%         sample_num = hog_sz(1) * hog_sz(2);
        
        %% car sample input
        [model.label, model.insc] = svm_hog(car.hog, hog_sz, model.label, model.insc, -1);
    end

    model.label = double(model.label);
    model.insc = double(model.insc);
    %input the label and hog，then output the parameters
    model.svm = svmtrain(model.label, model.insc, '-s 0 -t 0');
    svm = model.svm;
end

%% transformng label into proper format
function [label, insc] = svm_hog(hog, hog_sz, label, insc, x)
  %adjust the format of label and calculate insc
    sample_num = hog_sz(1) * hog_sz(2);
    % 正样本为1，负样本为-1
    label = [label; x];
    img.insc = [];
    % 按照列的方式从左到右、从上到下将HOG数据整理成一行，添加到总的HOG数据矩阵
    for sp_num = 1:sample_num
        for hog_num = 1:hog_sz(3)
            col = ceil(sp_num / hog_sz(1));
            row = sp_num - (col - 1) * hog_sz(1);
            img.insc = [img.insc, hog(row, col, hog_num)];
        end
    end
    insc = [insc; img.insc];
end

%% User-given database traing
function svm = CASE_train()
    c1_file_dir = 'pictures\train_case\answer1\';
    c2_file_dir = 'pictures\train_case\answer2\';
    negative_file_dir = 'pictures\NEGATIVE\';

    Readlist.c1_file = dir(fullfile(c1_file_dir,'*.jpg'));
    Readlist.c2_file = dir(fullfile(c2_file_dir,'*.jpg'));
    Readlist.Negative_file = dir(fullfile(negative_file_dir,'*.jpg'));

    model.label = [];
    model.insc = [];

    for i = 1:length(Readlist.c1_file)
        img_file = [c1_file_dir, Readlist.c1_file(i).name];
        
        im = imread(img_file);
        try
            img_single = im2single(rgb2gray(im));
        catch
            img_single = im2single(im);
        end
        img_car = img_single;
        car.img = imresize(img_car, [64, 64]);
        
        car.hog = vl_hog(car.img, 8);
        hog_sz = size(car.hog);
        
        %% car sample input
        [model.label, model.insc] = svm_hog(car.hog, hog_sz, model.label, model.insc, 1);
    end

    for i = 1:length(Readlist.Negative_file)
        img_file = [negative_file_dir, Readlist.Negative_file(i).name];
        
        im = imread(img_file);
        try
            img_single = im2single(rgb2gray(im));
        catch
            img_single = im2single(im);
        end
        img_car = img_single;
        car.img = imresize(img_car, [64, 64]);
        
        car.hog = vl_hog(car.img, 8);
        hog_sz = size(car.hog);
        
        %% car sample input
        [model.label, model.insc] = svm_hog(car.hog, hog_sz, model.label, model.insc, -1);
    end

    model.label = double(model.label);
    model.insc = double(model.insc);
    %input the label and hog，then output the parameters
    model.svm = svmtrain(model.label, model.insc, '-s 0 -t 0');
    svm = model.svm;

    % 计算整理第二个方向的正样本集，负样本集不变
    model.label = [];
    model.insc = [];

    for i = 1:length(Readlist.c2_file)
        img_file = [c2_file_dir, Readlist.c2_file(i).name];
        
        im = imread(img_file);
        try
            img_single = im2single(rgb2gray(im));
        catch
            img_single = im2single(im);
        end
        img_car = img_single;
        car.img = imresize(img_car, [64, 64]);
        
        car.hog = vl_hog(car.img, 8);
        hog_sz = size(car.hog);
%         sample_num = hog_sz(1) * hog_sz(2);
        
        %% car sample input
        [model.label, model.insc] = svm_hog(car.hog, hog_sz, model.label, model.insc, 1);
    end

    for i = 1:length(Readlist.Negative_file)
        img_file = [negative_file_dir, Readlist.Negative_file(i).name];
        
        im = imread(img_file);
        try
            img_single = im2single(rgb2gray(im));
        catch
            img_single = im2single(im);
        end
        img_car = img_single;
        car.img = imresize(img_car, [64, 64]);
        
        car.hog = vl_hog(car.img, 8);
        hog_sz = size(car.hog);
%         sample_num = hog_sz(1) * hog_sz(2);
        
        %% car sample input
        [model.label, model.insc] = svm_hog(car.hog, hog_sz, model.label, model.insc, -1);
    end

    model.label = double(model.label);
    model.insc = double(model.insc);
    %input the label and hog，then output the parameters
    model.svm = svmtrain(model.label, model.insc, '-s 0 -t 0');
    % 将所有训练出的不同的svm值被存进同一个数组（1xN矩阵）
    svm = [svm, model.svm];
end