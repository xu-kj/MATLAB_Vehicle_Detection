%% --------------------Main Function
function scan_result = testing()
    % Image scan and test
    load svmModel

    % ----------Parameters
    hog_sz            = [8, 8, 31];
    img_sz            = [64, 64];     % single scanning cell's size
    cell_move         = 0.2;          % every time, move the scanning box by 0.5 * side length
    r_print           = 1;            % 1 for printing the boxes, 0 for only calculating the result
    img_show          = 1;            % 1 for showing the image when it is read in
    expected_value    = 0.10;         % when decision value is larger than 0.1, the cell is judged as a car
    fgd_value         = 0.25;         % when the foreground shade portion 
    
    img_dir = 'pictures\test_case\2video\';
    fgd_dir = 'pictures\test_case\2foreground\';
    Readlist.img = dir(fullfile(img_dir,'*.jpg'));
    Readlist.fgd = dir(fullfile(fgd_dir,'*.jpg'));
    for m = 1:length(Readlist.img)
        img_file = [img_dir, Readlist.img(m).name];
        fgd_file = [fgd_dir, Readlist.fgd(m).name];
        [img_test, img_orig_sz] = ImgRead(img_file, img_show);
        fgd_file                = ImgRead(fgd_file, 0);
%         test.img = imresize(img_test, img_sz);

        %set threshold to remove some error
        scan_result = SVM_Scan(model.svm, img_test, fgd_file, ...
                               fgd_value, img_sz, img_orig_sz, hog_sz, ...
                               cell_move, expected_value, r_print);
        name_length = length(Readlist.img(m).name);
        saveas(gcf, ['output_0704_', sprintf('%04d', ...
               str2num(Readlist.img(m).name(7:name_length - 4)))], 'jpg');
%         print(gcf, '-dpng', 'print_method.png');
        
    end
end

%% --------------------Sub Functions
function [label, insc] = svm_hog(hog, hog_sz, label, insc, x)
    % Calculate [label, insc] for given HOG
    %adjust the format of label and calculate insc
    %hog: The hog calculated by vl_hog funtion
    %hog_sz: The size of hog above by function size()
    %label: The label for samples, and it is formed by x
    %x: 1 for positive samples
    %   -1 for nagative sample s
    sample_num = hog_sz(1) * hog_sz(2);
    label = [label; x];
    img.insc = [];
    for sp_num = 1:sample_num
        for hog_num = 1:hog_sz(3)
            col = ceil(sp_num / hog_sz(1));
            row = sp_num - (col - 1) * hog_sz(1);
            img.insc = [img.insc, hog(row, col, hog_num)];
        end
    end
    insc = [insc; img.insc];
end

function box = print_rect(i, j, cell_sz_1, cell_sz_2, cell_move, r_print)
    % Calculate absolute position of the box, and print it
    % i: loop viriable i from the main loop, shows the position of rect
    % j: loop viriable j from the main loop, shows the position of rect
    % cell_sz_1: the width of the rect
    % cell_sz_2: the height of the rect
    % cell_move: main parameter from mai function, =0.5
    % img_sz: the size of scanned image
    % img_orig_sz: the size of image before resize
    % r_print: 1 for printing the boxes, 0 for only calculating the result
    pos_1 = 1 + i * cell_sz_1 * cell_move;
    pos_2 = 1 + j * cell_sz_2 * cell_move;
    pos_1 = ceil(pos_1);
    pos_2 = ceil(pos_2);
    length_w = floor(cell_sz_2) - 1;
    length_h = floor(cell_sz_1) - 1;
    box = [pos_2, pos_1, length_w, length_h];
    if r_print == 1
        box_handle = rectangle('position', box);
        set(box_handle, 'edgecolor','y', 'linewidth',1);
    end
end

function [img_test ,img_orig_sz] = ImgRead(img_file, img_show)
    im = imread(img_file);
    if(img_show == 1)
      figure(1); imshow(im);
    end

    % convert the image to processable single matrix
    try
        img_test = im2single(rgb2gray(im));
    catch
        img_test = im2single(im);
    end
    img_orig_sz = size(img_test);
end

function result = SVM_Test(input_result, label, insc, svm, i, j, ...
    cell_sz_1, cell_sz_2, cell_move, exp_value, r_print)
    % label: the label of sample showing whether it is positive or nagetive
    % insc: stores all hog of samples
    % svm: model.svm from train
    % i: loop viriable i from the main loop, shows the position of rect
    % j: loop viriable j from the main loop, shows the position of rect
    % cell_sz_1: the width of the rect
    % cell_sz_2: the height of the rect
    % cell_move: main parameter from mai function, =0.5
    % img_sz: the size of scanned image
    % img_orig_sz: the size of image before resize
    % r_print: 1 for printing the boxes, 0 for only calculating the result
    result = input_result;
    for k = 1:length(svm)
        [predict_label, accuracy, decision_values] = ...
            svmpredict(label, insc, svm(k), '-b 0');
        if (accuracy(1) == 100) && (decision_values >= exp_value)
            box = print_rect(i, j, cell_sz_1, cell_sz_2, cell_move, r_print);
            result = [input_result; box];
            break;
        end
    end
end

function scan_result = SVM_Scan(svm, img, fgd, fgd_value, ...
    img_sz, img_orig_sz, hog_sz, cell_move, exp_value, r_print)
    % scan the image and show the boxes
    % svm: model.svm from train
    % img: tested image
    % img_sz: the size of scanned image
    % img_orig_sz: the size of image before resize
    % box_sz: imgsize(1), shows the size of the rectangular
    % hog_sz: main parameter, = [8, 8, 31]
    % cell_move: main parameter, = 0.5
    % exp_value: main parameter, =0.1
    % r_print: 1 for printing the boxes, 0 for only calculating the result 
    scan_result = [];
    box_sz_1 = floor(log2(img_orig_sz(1)));
    box_sz_2 = floor(log2(img_orig_sz(2)));
    for sz_num_1 = 6:box_sz_1
       for sz_num_2 = 6:box_sz_2
          cell_sz_1 = 2^sz_num_1; cell_sz_2 = 2^sz_num_2;
          scan_result = PyramidScan(scan_result, img, fgd, fgd_value, ...
              cell_sz_1, cell_sz_2, cell_move, img_sz, img_orig_sz, ...
              hog_sz, svm, exp_value, r_print);
       end
    end
end

function scan_result = PyramidScan(input_result, img, fgd, fgd_value, ...
    cell_sz_1, cell_sz_2, cell_move, img_sz, img_orig_sz, hog_sz, ...
    svm, exp_value, r_print)
  % scan with various size of boxes    
  % img: tested image
  % cell_sz_1: the width of the rect
  % cell_sz_2: the height of the rect
  % cell_move: main parameter, = 0.5
  % box_sz: imgsize(1), shows the size of the rectangular
  % img_sz: the size of scanned image
  % img_orig_sz: the size of image before resize
  % hog_sz: main parameter, = [8, 8, 31]
  % svm: model.svm from train
  % input_result: the output of SVM_Scan
  % exp_value: main parameter, =0.1
  % r_print: 1 for printing the boxes, 0 for only calculating the result
    for i = 0:floor((img_orig_sz(1) - cell_sz_1)/(cell_sz_1 * cell_move))
        for j = 0:floor((img_orig_sz(2) - cell_sz_2)/(cell_sz_2 * cell_move))
            step_1 = floor(cell_sz_1 * cell_move); 
            step_2 = floor(cell_sz_2 * cell_move);
            cell.fgd = fgd((1+i*step_1):(cell_sz_1+i*step_1),...
                (1+j*step_2):(cell_sz_2+j*step_2));
            if FGD_check(cell.fgd, fgd_value)
                cell.img = img((1+i*step_1):(cell_sz_1+i*step_1),...
                    (1+j*step_2):(cell_sz_2+j*step_2));
                cell.img = imresize(cell.img, img_sz);
                cell.hog = vl_hog(cell.img, 8);
                cell.label = []; cell.insc = [];
                [cell.label, cell.insc] = svm_hog(cell.hog, hog_sz, ...
                    cell.label, cell.insc, 1);
                cell.label = double(cell.label); 
                cell.insc = double(cell.insc);
                input_result = SVM_Test(input_result, cell.label, ...
                    cell.insc, svm, i, j, cell_sz_1, cell_sz_2, ...
                    cell_move, exp_value, r_print);
            end
        end
    end
    scan_result = input_result;
end

function scan_check = FGD_check(fgd, fgd_value)
    target = sum(sum(fgd));
    total  = prod(size(fgd));
    ratio  = target / total;
    if ratio >= fgd_value
        scan_check = 1;
    else
        scan_check = 0;
    end
end
