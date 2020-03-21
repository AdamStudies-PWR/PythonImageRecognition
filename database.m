clc;
clear;
load("wiki.mat");
[age,~]=datevec(datenum(wiki.photo_taken,7,1)-wiki.dob);
warning ('off','all');
mkdir ('Croped');
a = 0;
T = [];
paths = [];

for k = 1 : length(wiki.face_score)
    if ((wiki.face_score(k) > 5.283) && isnan(wiki.second_face_score(k)) && !isnan(wiki.gender(k)))
        a = a + 1;
        T = [T, k];
        paths = [paths, wiki.full_path(k)];
        %temp = imcrop(imread(char(wiki.full_path(k))),[wiki.face_location(1, 2) wiki.face_location(1, 4) wiki.face_location(1, 1) wiki.face_location(1, 3)]);
        temp = imread(char(wiki.full_path(k)));
        %temp(wiki.face_location(k,2):wiki.face_location(k,4),wiki.face_location(k,1):wiki.face_location(k, 3));
        %image(temp)
        name  = ['Croped/' num2str(a) 'g' num2str(wiki.gender(k)) 'a' num2str(age(k)) '.png'];
        imwrite(temp,  name);
    end
end

disp(a );

tttt = imread( char(paths(6)) );
image(tttt);