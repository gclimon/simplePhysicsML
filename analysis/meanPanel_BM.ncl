;================================================;
;	Example ncl script to produce the set of
;       error time series for ML algorithms
; Created by Christiane Jablonowski & Garrett Limon
;	(University of Michigan)
;
;================================================;
load "$NCARG_ROOT/lib/ncarg/nclscripts/csm/gsn_code.ncl"   
load "$NCARG_ROOT/lib/ncarg/nclscripts/csm/gsn_csm.ncl"
load "$NCARG_ROOT/lib/ncarg/nclscripts/csm/contributed.ncl"   
load "$NCARG_ROOT/lib/ncarg/nclscripts/csm/shea_util.ncl"   
; ================================================;

begin

;=================================================;
; open file and read in data 
; GIVEN AS (time,lev,lat,lon) from 0 to end_time-1
;=================================================;

  type = "TJ_BM"
  dir = "/glade/scratch/glimon/TJ_19_convection/70_4hr/"

  pi    = 4.*atan(1.)

  fname1 = "TJ_19_convection.cam.h0.0010-12-22-00000.nc"
  fname2 = "TJ_19_convection.cam.h1.0010-12-22-00000.nc"
  data1 = dir+fname1
  data2 = dir+fname2
  f1 = addfile(data1,"r")
  f2 = addfile(data2,"r")
  lat  = f1->lat
  lon  = f1->lon

  T = f1->T(:,:,:,:)
  Qm = f1->Q(:,:,:,:)
  RH = f1->RELHUM(:,:,:,:)
  U = f1->U(:,:,:,:)
  dTm = f2->PTTEND(:,:,:,:)
  dQm = f2->PTEQ(:,:,:,:)
  PLm = f2->PRECL(:,:,:)
  PCm = f2->PRECC(:,:,:)
  PTm = PLm + PCm

  dT = dTm*86400.
  copy_VarMeta(dTm,dT)
  Q = Qm*1000.
  copy_VarMeta(Qm,Q)
  dQ = dQm*1000*86400
  copy_VarMeta(dQm,dQ)
  PT = PTm*1000*86400
  copy_VarMeta(PTm,PT)
  PL = PLm*1000*86400
  copy_VarMeta(PLm,PL)
  PC = PCm*1000*86400
  copy_VarMeta(PCm,PC)


;************************************************
; time average
;************************************************
  T_tm = dim_avg_n_Wrap(T,0)
  Q_tm = dim_avg_n_Wrap(Q,0)
  RH_tm = dim_avg_n_Wrap(RH,0)
  U_tm = dim_avg_n_Wrap(U,0)
  dT_tm = dim_avg_n_Wrap(dT,0)
  dQ_tm = dim_avg_n_Wrap(dQ,0)
  PL_tm = dim_avg_n_Wrap(PL,0)
  PC_tm = dim_avg_n_Wrap(PC,0)
  PT_tm = dim_avg_n_Wrap(PT,0)

;************************************************
; zonal average
;************************************************
  T_m = dim_avg_n_Wrap(T_tm,2)
  Q_m = dim_avg_n_Wrap(Q_tm,2)
  RH_m = dim_avg_n_Wrap(RH_tm,2)
  U_m = dim_avg_n_Wrap(U_tm,2)
  dT_m = dim_avg_n_Wrap(dT_tm,2)
  dQ_m = dim_avg_n_Wrap(dQ_tm,2)
  PL_m = dim_avg_n_Wrap(PL_tm,1)
  PC_m = dim_avg_n_Wrap(PC_tm,1)
  PT_m = dim_avg_n_Wrap(PT_tm,1)

  pltType = "eps"
  panelSize = (/3,3/)    ; panel dimensions

  wks  = gsn_open_wks(pltType,type)
  gsn_define_colormap(wks,"WhViBlGrYeOrRe")

plot = new(9,graphic)

	res1                      = True
	res1@gsnDraw              = False          		; panel plot 
	res1@gsnFrame             = False			; don't draw yet
	res1@cnFillOn             = True
	res1@cnLinesOn            = False
	res1@gsnSpreadColors      = True
	res1@lbLabelAutoStride    = True
;	res1@tiMainString         = ""
	res1@vpWidthF             = 0.5                         ; set aspect ratio
	res1@vpHeightF            = 0.32                        ; set aspect ratio
	res1@cnInfoLabelOn        = False			; don't give each
	res1@cnLineLabelsOn       = False			; panel its own
	res1@lbLabelBarOn         = True			; label bar
	res1@gsnMaximize          = True  
	res1@trYReverse           = True      			; reverses y-axis, false
;       res1@tmXBMinorPerMajor    = 4                           ; no effect in the current setup
        res1@gsnYAxisIrregular2Linear = True   ; Convert Y axis to linear

	pres                          = True
	pres@gsnMaximize              = True  
	pres@gsnPanelLabelBar         = False			; Communal label bar
        pres@gsnPanelBottom           = 0.			 ; shrink panel plot by setting bottom edge of plot
        pres@gsnPanelTop              = 0.96			 ; shrink panel plot by setting top edge of plot
;	pres@gsnPanelLeft             = 0.1
;	pres@gsnPanelRight            = 0.9
	pres@pmLabelBarOrthogonalPosF = -0.01            ; negative moves down like -0.03
        pres@pmLabelBarWidthF         = 0.4
	pres@gsnFrame                 = False
	pres@lbLabelStride            = 2
        pres@lbTitleFontHeightF       = 0.06
        pres@gsnPanelYWhiteSpacePercent = 3
        pres@gsnPanelXWhiteSpacePercent = 2
        pres@lbLabelFontHeightF       = 0.01
        pres@gsnPanelFigureStrings    = (/"a)","b)","c)","d)","e)","f)","g)","h)","i)"/) ; add strings to panel
        pres@gsnPanelFigureStringsFontHeightF = 0.01
        pres@amJust                   = "TopLeft"
        pres@gsnAttachBorderOn        = "False"
        pres@gsnPanelMainFontHeightF  = 0.015

	res1@cnLevelSelectionMode = "AutomaticLevels" 
        ; res1@cnMaxLevelValF  =   8.0                 ; max contour color label
        ; res1@cnMinLevelValF  =   -2.4                  ; min contour color label
        ; res1@cnLevelSpacingF = 0.75                         ; contour color spacing

	res1@gsnRightString = "K"
	res1@tiYAxisString   = "Pressure (hPa)"
        plot(0) = gsn_csm_contour(wks,T_m(:,:),res1)	; plot var

	res1@gsnRightString = "g/kg"
        plot(1) = gsn_csm_contour(wks,Q_m(:,:),res1)	; plot var

	res1@gsnRightString = "%"
        plot(2) = gsn_csm_contour(wks,RH_m(:,:),res1)	; plot var

	res1@gsnRightString = "K/day"
        plot(3) = gsn_csm_contour(wks,dT_m(:,:),res1)	; plot var

	res1@gsnRightString = "g/kg/day"
        plot(4) = gsn_csm_contour(wks,dQ_m(:,:),res1)	; plot var

	res1@gsnRightString   = "m/s"
        plot(5) = gsn_csm_contour(wks,U_m(:,:),res1)	; plot var

	res1@trYReverse           = False      			; reverses y-axis, false

	; HOW TO SET Y LIMITS THE SAME

	res1@gsnRightString = ""
	res1@tiYAxisString   = "Precip Rate (mm/day)"
	res1@gsnLeftString = "Large-Scale Precip"
        plot(6) = gsn_csm_xy(wks,lat,PL_m(:),res1)	; plot var

	res1@gsnLeftString = "Convective Precip"
        plot(7) = gsn_csm_xy(wks,lat,PC_m(:),res1)	; plot var

	res1@gsnLeftString   = "Total Precip"
        plot(8) = gsn_csm_xy(wks,lat,PT_m(:),res1)	; plot var

	gsn_panel(wks,plot,panelSize,pres)			; panel plot
	frame(wks)

end
