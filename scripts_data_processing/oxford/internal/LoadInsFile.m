function ins_data = LoadInsFile(fname)

ins_file_id = fopen(fname);
headers = textscan(ins_file_id, '%s', 15, 'Delimiter',',');
ins_data = textscan(ins_file_id, ...
  '%u64 %s %f %f %f %f %f %f %s %f %f %f %f %f %f','Delimiter',',');
fclose(ins_file_id);