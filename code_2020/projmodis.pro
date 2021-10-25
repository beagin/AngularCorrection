function hdf_sd_varread,hdfid,varname,_extra = extra_keywords

  ;- Check arguments
  if (n_params() ne 2) then $
    message, 'Usage: HDF_SD_VARREAD, HDFID, VARNAME, DATA'
  if (n_elements(hdfid) eq 0) then $
    message, 'Argument HDFID is undefined'
  if (n_elements(varname) eq 0) then $
    message, 'Argument VARNAME is undefined'
  ;if (arg_present(data) eq 0) then $
  ;  message, 'Argument DATA cannot be modified'

  ;- Get index of the requested variable

  index = hdf_sd_nametoindex(hdfid, varname)

  if (index lt 0) then $
    message, 'SDS was not found: ' + varname

  ;- Select and read the SDS

  varid = hdf_sd_select(hdfid, index)

  hdf_sd_getdata, varid, data, _extra=extra_keywords

  hdf_sd_endaccess,varid


  return,data

end

;�����ļ�����������Щ
function hdf_sd_vardir,hdfid

  ;- Check arguments
  if (n_params() ne 1) then $
    message, 'Usage: RESULT = HDF_SD_VARDIR(HDFID)'
  if (n_elements(hdfid) eq 0) then $
    message, 'HDFID is undefined'

  ;- Set default return value
  varnames = ''

  ;- Get file information
  hdf_sd_fileinfo, hdfid, nvars, ngatts

  ;- If variables were found, get variable names
  if (nvars gt 0) then begin
    varnames = strarr(nvars)
    for index = 0L, nvars - 1L do begin
      varid = hdf_sd_select(hdfid, index)
      hdf_sd_getinfo, varid, name=name
      hdf_sd_endaccess, varid
      varnames[index] = name
    endfor
  endif

  ;- Return the result
  return, varnames

END




PRO ProjMODIS


  COMPILE_OPT idl2
  ENVI,/restore_base_save_files
  ENVI_BATCH_INIT, log_file='batch.txt'

  cd,'C:\Users\Dongjj\Desktop'
  
  
  filename = 'MYD11_L2.A2019346.1755.006.2019351215528.hdf'
  
  
  
HDFFile = hdf_sd_start(filename)
  
  HDFFile_att = hdf_sd_vardir(HDFFile) ;Ѱ������

;  print,HDFFile_att


  Lon = hdf_sd_varread(HDFFile,'Longitude');
  Lat = hdf_sd_varread(HDFfile,'Latitude')
  LST = hdf_sd_varread(HDFfile,'LST')*0.02

  
  
  hdf_sd_end,HDFfile
  
  
  ENVI_WRITE_ENVI_FILE,LST,r_fid=fid1,out_name= 'tempdata.dat'
  envi_file_query, fid1, nb = nb, dims = dim
  
  
  sizeofdata = size(lon,/dimensions)
  ns = sizeofdata[0]
  nl = sizeofdata[1]
  
  Count = 0L

  GCP = FLTARR(4,ns * nl)
  FOR NX =0,ns -1 DO BEGIN
    FOR NY = 0, nl -1 DO BEGIN

      GCP[0,Count] = LON[NX,NY]
      GCP[1,Count] = Lat[NX,NY]
      GCP[2,Count] = 5*NX+2
      GCP[3,Count] = 5*NY+2
      Count++
    ENDFOR
  ENDFOR
  
  
  RegFileName = strmid(filename,0,27)+"_LST_REG_geo.dat"

  pixel_size= [0.01,0.01]
  
  pos = [0]
  
  oproj = ENVI_proj_create(/geographic)

  ENVI_DOIT, 'envi_register_doit', w_fid=fid1, w_pos=POS, w_dims=dims, method=0,$
    out_name=RegFileName,pts=gcp, proj=oproj,r_fid=r_fid1, pixel_size=pixel_size, X0=MIN(gcp[0,*]),$
    Y0=MAX(gcp[1,*]), XSIZE=MAX(gcp[0,*])-MIN(gcp[0,*]), YSIZE=MAX(gcp[1,*])-MIN(gcp[1,*])
    
  ENVI_FILE_MNG,id=fid1,/remove,/delete   
  ENVI_BATCH_EXIT

  PRINT,"DONE"

END