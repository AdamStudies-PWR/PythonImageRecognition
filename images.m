function images()
  warning ('off','all');
  wiki = load('wiki_crop/wiki.mat');
  wiki
  #mkdir database;
  
  img(face_location(2):face_location(4),face_location(1):face_location(3),:)
  [age,~]=datevec(datenum(wiki.photo_taken,7,1)-wiki.dob); 
endfunction
