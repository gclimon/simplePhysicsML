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
  dry = path+"HS/"
  moist = path+"TJ/"
  conv = path+"TJBM/"
  fname = "results_50trees.nc"
  
  name = "dTdt_panel"

  ddT_f = addfile(dry + "PTTEND/rf/"+fname,"r")
  mdT_f = addfile(moist + "PTTEND/rf/results_aug.nc","r")
  cdT_f = addfile(conv + "PTTEND/rf/"+fname,"r")

; output format
;  pltType = "X11"
;  pltType = "png"
  pltType = "eps"
  panelSize = (/3,3/)    ; panel dimensions
  version   = ""

  lat  = mdT_f->lat
  lon  = mdT_f->lon
  pi    = 4.*atan(1.)
  lev  = mdT_f->lev

; choose the variable
  ddT = ddT_f->PTTEND(:,:,:,:)
  ddT_ml = ddT_f->PTTEND_ML_predicted(:,:,:,:)
  ddT_diff = ddT_f->PTTEND_difference(:,:,:,:)
  mdT = mdT_f->PTTEND(:,:,:,:)
  mdT_ml = mdT_f->PTTEND_ML_predicted(:,:,:,:)
  mdT_diff = mdT_f->PTTEND_difference(:,:,:,:)
  cdT = cdT_f->PTTEND(:,:,:,:)
  cdT_ml = cdT_f->PTTEND_ML_predicted(:,:,:,:)
  cdT_diff = cdT_f->PTTEND_difference(:,:,:,:)


  ddT_mod = ddT * 86400.
  copy_VarMeta(ddT,ddT_mod)
  ddT_ml_mod = ddT_ml * 86400.
  copy_VarMeta(ddT_ml,ddT_ml_mod)
  ddT_diff_mod = ddT_diff * 86400.  ; ML-CAM
  copy_VarMeta(ddT_diff,ddT_diff_mod)

  mdT_mod = mdT * 86400.
  copy_VarMeta(mdT,mdT_mod)
  mdT_ml_mod = mdT_ml * 86400.
  copy_VarMeta(mdT_ml,mdT_ml_mod)
  mdT_diff_mod = mdT_diff * 86400.  ; ML-CAM
  copy_VarMeta(mdT_diff,mdT_diff_mod)

  cdT_mod = cdT * 86400.
  copy_VarMeta(cdT,cdT_mod)
  cdT_ml_mod = cdT_ml * 86400.
  copy_VarMeta(cdT_ml,cdT_ml_mod)
  cdT_diff_mod = cdT_diff * 86400.  ; ML-CAM
  copy_VarMeta(cdT_diff,cdT_diff_mod)


;************************************************
; zonal average
;************************************************

  ddT_zm = dim_avg_n_Wrap(ddT_mod,3)
  ddT_ml_zm  = dim_avg_n_Wrap(ddT_ml_mod,3)
  ddT_diff_zm  = dim_avg_n_Wrap(ddT_diff_mod,3)

  mdT_zm = dim_avg_n_Wrap(mdT_mod,3)
  mdT_ml_zm  = dim_avg_n_Wrap(mdT_ml_mod,3)
  mdT_diff_zm  = dim_avg_n_Wrap(mdT_diff_mod,3)

  cdT_zm = dim_avg_n_Wrap(cdT_mod,3)
  cdT_ml_zm  = dim_avg_n_Wrap(cdT_ml_mod,3)
  cdT_diff_zm  = dim_avg_n_Wrap(cdT_diff_mod,3)


;************************************************
; time average
;************************************************

  ddT_m = dim_avg_n_Wrap(ddT_zm,0)
  ddT_ml_m  = dim_avg_n_Wrap(ddT_ml_zm,0)
  ddT_diff_m  = dim_avg_n_Wrap(ddT_diff_zm,0)

  mdT_m = dim_avg_n_Wrap(mdT_zm,0)
  mdT_ml_m  = dim_avg_n_Wrap(mdT_ml_zm,0)
  mdT_diff_m  = dim_avg_n_Wrap(mdT_diff_zm,0)

  cdT_m = dim_avg_n_Wrap(cdT_zm,0)
  cdT_ml_m  = dim_avg_n_Wrap(cdT_ml_zm,0)
  cdT_diff_m  = dim_avg_n_Wrap(cdT_diff_zm,0)

printMinMax(ddT_m,True)
printMinMax(ddT_ml_m,True)
printMinMax(ddT_diff_m,True)

printMinMax(mdT_m,True)
printMinMax(mdT_ml_m,True)
printMinMax(mdT_diff_m,True)

printMinMax(cdT_m,True)
printMinMax(cdT_ml_m,True)
printMinMax(cdT_diff_m,True)

  wks  = gsn_open_wks(pltType,name)
  gsn_define_colormap(wks,"WhViBlGrYeOrReWh")

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
;        pres@gsnPanelMainString       = "Moist dT/dt Field"  ; Adjust name!
        pres@gsnPanelMainFontHeightF  = 0.015

	res1@cnLevelSelectionMode = "ManualLevels" 
	; res1@cnLevelSelectionMode = "ExplicitLevels" 

	res1@gsnRightString = "K/day"
	res1@tiYAxisString   = "Pressure (hPa)"

	res1@gsnCenterString = "CAM"
        ; res1@cnLevels = (/-0.6, -0.3, 0.0, 0.3, 0.6, 0.9, 1.2, 1.8, 2.4, 2.8/)
	; res1@lbLabelStrings = (/"-0.6", "-0.3", "0.0", "0.3", "0.6", "0.9", "1.2", "1.8", "2.4", "2.8"/)

        res1@cnMaxLevelValF  =   2.0                 ; max contour color label
        res1@cnMinLevelValF  =   -2.0                  ; min contour color label
        res1@cnLevelSpacingF = 0.25                         ; contour color spacing

	res1@gsnLeftString = "Dry dT/dt"
        plot(0) = gsn_csm_contour(wks,ddT_m(:,:),res1)	; plot var

	res1@gsnCenterString = "ML"
	plot(1) = gsn_csm_contour(wks,ddT_ml_m(:,:),res1)	; plot var

	res1@gsnCenterString = "ML - CAM"

	; res1@cnLevels = (/-0.015, -0.012, -0.009, -0.006, -0.003, 0.0, 0.003, 0.006, 0.009, 0.012/)	
	; res1@lbLabelStrings = (/"-0.015", "-0.012", "-0.009", "-0.006", "-0.003", "0.0", "0.003", "0.006", "0.009", "0.012"/)	
        res1@cnMaxLevelValF  =    0.01                 ; max contour color label
        res1@cnMinLevelValF  =   -0.01                ; min contour color label
        res1@cnLevelSpacingF = 0.002                           ; contour color spacing

	plot(2) = gsn_csm_contour(wks,ddT_diff_m(:,:),res1)	; plot var

;;;;;;;

	; res1@cnLevels = (/-2.0, -1.0, 0.0, 1.5, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0/)	
	; res1@lbLabelStrings = (/"-2.0", "-1.0", "0.0", "1.5", "3.0", "4.0", "5.0", "6.0", "7.0", "8.0"/)	
        res1@cnMaxLevelValF  =    6.                 ; max contour color label
        res1@cnMinLevelValF  =   -6.                  ; min contour color label
        res1@cnLevelSpacingF = 1.0                         ; contour color spacing

	res1@gsnCenterString = "CAM"
	res1@gsnLeftString = "Moist dT/dt"
        plot(3) = gsn_csm_contour(wks,mdT_m(:,:),res1)	; plot var

	res1@gsnCenterString = "ML"
	plot(4) = gsn_csm_contour(wks,mdT_ml_m(:,:),res1)	; plot var

	res1@gsnCenterString = "ML - CAM"

	; res1@cnLevels = (/-0.08, -0.06, -0.04, -0.02, 0.0, 0.02, 0.04, 0.07, 0.10, 0.12/)	
	; res1@lbLabelStrings = (/"-0.08", "-0.06", "-0.04", "-0.02", "0.0", "0.02", "0.04", "0.07", "0.10", "0.12"/)		
        res1@cnMaxLevelValF  =    0.05                 ; max contour color label
        res1@cnMinLevelValF  =   -0.05                 ; min contour color label
        res1@cnLevelSpacingF = 0.01                           ; contour color spacing

	plot(5) = gsn_csm_contour(wks,mdT_diff_m(:,:),res1)	; plot var

;;;;;;;

	; res1@cnLevels = (/-2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0, 2.3/)
	; res1@lbLabelStrings = (/"-2.0", "-1.5", "-1.0", "-0.5", "0.0", "0.5", "1.0", "1.5", "2.0", "2.3"/)
        res1@cnMaxLevelValF  =   2.0                 ; max contour color label
        res1@cnMinLevelValF  =   -2.0                  ; min contour color label
        res1@cnLevelSpacingF = 0.5                         ; contour color spacing
	res1@gsnCenterString = "CAM"
	res1@gsnLeftString = "Conv. dT/dt"
        plot(6) = gsn_csm_contour(wks,cdT_m(:,:),res1)	; plot var

	res1@gsnCenterString = "ML"
	plot(7) = gsn_csm_contour(wks,cdT_ml_m(:,:),res1)	; plot var

	res1@gsnCenterString = "ML - CAM"

	; res1@cnLevels = (/-0.05, -0.03, -0.01, -0.005, 0.0, 0.005, 0.01, 0.03, 0.05, 0.06/)	
	; res1@lbLabelStrings = (/"-0.05", "-0.03", "-0.01", "-0.005", "0.0", "0.005", "0.01", "0.03", "0.05", "0.06"/)	
        res1@cnMaxLevelValF  =    0.05                 ; max contour color label
        res1@cnMinLevelValF  =   -0.05                 ; min contour color label
        res1@cnLevelSpacingF =    0.01                           ; contour color spacing

	plot(8) = gsn_csm_contour(wks,cdT_diff_m(:,:),res1)	; plot var

	gsn_panel(wks,plot,panelSize,pres)			; panel plot
	frame(wks)

	; if (type .ne. "X11") then
    	; system("convert -trim "+plotname+"."+pfmt+" "+plotname+"."+pfmt)
  	; end if

end
