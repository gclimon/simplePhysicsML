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



  path = "/glade/scratch/glimon/simplePhysML/19/"
  moist = path+"TJ/"
  conv = path+"TJBM/"
  fname = "results.nc"
  mPL_f = addfile(moist + "PRECL/rf/results_aug.nc","r")
  cPL_f = addfile(conv + "PRECL/rf/"+fname,"r")
  cPC_f = addfile(conv + "PRECC/rf/"+fname,"r")

  name = "PREC_panel"

; output format
;  pltType = "X11"
;  pltType = "png"
  pltType = "eps"
  panelSize = (/2,2/)    ; panel dimensions

  lat  = mPL_f->lat
  lon  = mPL_f->lon
  pi    = 4.*atan(1.)

; choose the variable
  mPL = mPL_f->PRECL(:,:,:)
  mPL_ml = mPL_f->PRECL_ML_predicted(:,:,:)
  mPL_diff = mPL_f->PRECL_difference(:,:,:)
  cPL = cPL_f->PRECL(:,:,:)
  cPL_ml = cPL_f->PRECL_ML_predicted(:,:,:)
  cPL_diff = cPL_f->PRECL_difference(:,:,:)
  cPC = cPC_f->PRECC(:,:,:)
  cPC_ml = cPC_f->PRECC_ML_predicted(:,:,:)
  cPC_diff = cPC_f->PRECC_difference(:,:,:)

  mPL_mod = mPL * 86400. * 1000.
  copy_VarMeta(mPL,mPL_mod)
  mPL_ml_mod = mPL_ml * 86400. * 1000.
  copy_VarMeta(mPL_ml,mPL_ml_mod)
  mPL_diff_mod = mPL_diff * 86400. * 1000.  ; ML-CAM
  copy_VarMeta(mPL_diff,mPL_diff_mod)

  cPL_mod = cPL * 86400. * 1000.
  copy_VarMeta(cPL,cPL_mod)
  cPL_ml_mod = cPL_ml * 86400. * 1000.
  copy_VarMeta(cPL_ml,cPL_ml_mod)
  cPL_diff_mod = cPL_diff * 86400. * 1000.  ; ML-CAM
  copy_VarMeta(cPL_diff,cPL_diff_mod)

  cPC_mod = cPC * 86400. * 1000.
  copy_VarMeta(cPC,cPC_mod)
  cPC_ml_mod = cPC_ml * 86400. * 1000.
  copy_VarMeta(cPC_ml,cPC_ml_mod)
  cPC_diff_mod = cPC_diff * 86400. * 1000.  ; ML-CAM
  copy_VarMeta(cPC_diff,cPC_diff_mod)


;************************************************
; zonal average
;************************************************

  mPL_zm = dim_avg_n_Wrap(mPL_mod,2)
  mPL_ml_zm  = dim_avg_n_Wrap(mPL_ml_mod,2)
  mPL_diff_zm  = dim_avg_n_Wrap(mPL_diff_mod,2)

  cPL_zm = dim_avg_n_Wrap(cPL_mod,2)
  cPL_ml_zm  = dim_avg_n_Wrap(cPL_ml_mod,2)
  cPL_diff_zm  = dim_avg_n_Wrap(cPL_diff_mod,2)

  cPC_zm = dim_avg_n_Wrap(cPC_mod,2)
  cPC_ml_zm  = dim_avg_n_Wrap(cPC_ml_mod,2)
  cPC_diff_zm  = dim_avg_n_Wrap(cPC_diff_mod,2)


;************************************************
; time average
;************************************************

  mPL_m = dim_avg_n_Wrap(mPL_zm,0)
  mPL_ml_m  = dim_avg_n_Wrap(mPL_ml_zm,0)
  mPL_diff_m  = dim_avg_n_Wrap(mPL_diff_zm,0)

  cPL_m = dim_avg_n_Wrap(cPL_zm,0)
  cPL_ml_m  = dim_avg_n_Wrap(cPL_ml_zm,0)
  cPL_diff_m  = dim_avg_n_Wrap(cPL_diff_zm,0)

  cPC_m = dim_avg_n_Wrap(cPC_zm,0)
  cPC_ml_m  = dim_avg_n_Wrap(cPC_ml_zm,0)
  cPC_diff_m  = dim_avg_n_Wrap(cPC_diff_zm,0)


  plt_mPL = new((/2,dimsizes(mPL_m)/),double)
  plt_mPL(0,:) = mPL_m
  plt_mPL(1,:) = mPL_ml_m

  plt_cPL = new((/2,dimsizes(cPL_m)/),double)
  plt_cPL(0,:) = cPL_m
  plt_cPL(1,:) = cPL_ml_m

  plt_cPC = new((/2,dimsizes(cPC_m)/),double)
  plt_cPC(0,:) = cPC_m
  plt_cPC(1,:) = cPC_ml_m

  wks  = gsn_open_wks(pltType,name)
  gsn_define_colormap(wks,"WhViBlGrYeOrRe")

plot = new(4,graphic)

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
	res1@trYReverse           = False      			; reverses y-axis, false
;       res1@tmXBMinorPerMajor    = 4                           ; no effect in the current setup
        res1@gsnYAxisIrregular2Linear = True   ; Convert Y axis to linear

	pres                          = True
	pres@gsnMaximize              = True  
	pres@gsnPanelLabelBar         = False			; Communal label bar
;        pres@gsnPanelBottom           = 0.			 ; shrink panel plot by setting bottom edge of plot
 ;       pres@gsnPanelTop              = 0.96			 ; shrink panel plot by setting top edge of plot
;	pres@gsnPanelLeft             = 0.1
;	pres@gsnPanelRight            = 0.9
	pres@pmLabelBarOrthogonalPosF = -0.01            ; negative moves down like -0.03
        pres@pmLabelBarWidthF         = 0.4
	pres@gsnFrame                 = False
	pres@lbLabelStride            = 2
        pres@lbTitleFontHeightF       = 0.06
  ;      pres@gsnPanelYWhiteSpacePercent = 3
   ;     pres@gsnPanelXWhiteSpacePercent = 2
        pres@lbLabelFontHeightF       = 0.01
        pres@gsnPanelFigureStrings    = (/"a)","b)","c)","c)","e)","f)"/) ; add strings to panel
        pres@gsnPanelFigureStringsFontHeightF = 0.01
        pres@amJust                   = "TopLeft"
        pres@gsnAttachBorderOn        = "False"
;        pres@gsnPanelMainString       = "Moist dT/dt Field"  ; Adjust name!
        pres@gsnPanelMainFontHeightF  = 0.015

	; res1@cnLevelSelectionMode = "ManualLevels" 

        ; res1@cnMaxLevelValF  =   26.0                 ; max contour color label
        ; res1@cnMinLevelValF  =   0.015                  ; min contour color label
        ; res1@cnLevelSpacingF = 2.0                         ; contour color spacing
	res1@gsnRightString = ""

	; res1@tiYAxisString   = "Pressure (hPa)"

  res1@pmLegendDisplayMode    = "Always"            ; turn on legend
  res1@pmLegendSide           = "Top"               ; Change location of 
  res1@pmLegendParallelPosF   = 0.9                  ; move units right
  res1@pmLegendOrthogonalPosF = -0.25                ; move units down
  res1@pmLegendWidthF         = 0.015                ; Change width and
  res1@pmLegendHeightF        = 0.05                ; height of legend.
  res1@lgPerimOn              = False               ; turn off box around
  res1@lgLabelFontHeightF     = .02                 ; label font height
  res1@xyExplicitLegendLabels = (/"  CAM","  ML"/)         ; create explicit labels

	res1@xyLineThicknesses = (/  3.0,   3.0/)          ; make second line thicker
 	res1@xyLineColors      = (/"blue","orange"/)          ; change line color

	res1@gsnCenterString = "Moist Large-Scale Precip."
	res1@tiYAxisString   = "Precip Rate (mm/day)"
        plot(0) = gsn_csm_xy(wks,lat,plt_mPL,res1)	; plot mPL	

	res1@gsnCenterString = "Convection Large-Scale Precip."
        plot(1) = gsn_csm_xy(wks,lat,plt_cPL,res1)	; plot mPL	

	res1@gsnCenterString = "Convection Convective Precip."
        plot(3) = gsn_csm_xy(wks,lat,plt_cPC,res1)	; plot mPL	

	gsn_panel(wks,plot,panelSize,pres)			; panel plot
	frame(wks)
end
