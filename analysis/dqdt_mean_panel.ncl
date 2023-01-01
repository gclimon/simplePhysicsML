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
  
  name = "dQdt_panel"

  mdQ_f = addfile(moist + "PTEQ/rf/results_aug.nc","r")
  cdQ_f = addfile(conv + "PTEQ/rf/"+fname,"r")

; output format
  pltType = "eps"
  panelSize = (/2,3/)    ; panel dimensions

  lat  = mdQ_f->lat
  lon  = mdQ_f->lon
  pi    = 4.*atan(1.)
  lev  = mdQ_f->lev

; choose the variable
  mdQ = mdQ_f->PTEQ(:,:,:,:)
  mdQ_ml = mdQ_f->PTEQ_ML_predicted(:,:,:,:)
  mdQ_diff = mdQ_f->PTEQ_difference(:,:,:,:)
  cdQ = cdQ_f->PTEQ(:,:,:,:)
  cdQ_ml = cdQ_f->PTEQ_ML_predicted(:,:,:,:)
  cdQ_diff = cdQ_f->PTEQ_difference(:,:,:,:)

; convert units and reserve attributes
  mdQ_mod = mdQ * 86400. * 1000.
  copy_VarMeta(mdQ,mdQ_mod)
  mdQ_ml_mod = mdQ_ml * 86400. * 1000.
  copy_VarMeta(mdQ_ml,mdQ_ml_mod)
  mdQ_diff_mod = mdQ_diff * 86400. * 1000.  ; ML-CAM
  copy_VarMeta(mdQ_diff,mdQ_diff_mod)

  cdQ_mod = cdQ * 86400. * 1000.
  copy_VarMeta(cdQ,cdQ_mod)
  cdQ_ml_mod = cdQ_ml * 86400. * 1000.
  copy_VarMeta(cdQ_ml,cdQ_ml_mod)
  cdQ_diff_mod = cdQ_diff * 86400. * 1000.  ; ML-CAM
  copy_VarMeta(cdQ_diff,cdQ_diff_mod)


;************************************************
; zonal average
;************************************************

  mdQ_zm = dim_avg_n_Wrap(mdQ_mod,3)
  mdQ_ml_zm  = dim_avg_n_Wrap(mdQ_ml_mod,3)
  mdQ_diff_zm  = dim_avg_n_Wrap(mdQ_diff_mod,3)

  cdQ_zm = dim_avg_n_Wrap(cdQ_mod,3)
  cdQ_ml_zm  = dim_avg_n_Wrap(cdQ_ml_mod,3)
  cdQ_diff_zm  = dim_avg_n_Wrap(cdQ_diff_mod,3)


;************************************************
; time average
;************************************************

  mdQ_m = dim_avg_n_Wrap(mdQ_zm,0)
  mdQ_ml_m  = dim_avg_n_Wrap(mdQ_ml_zm,0)
  mdQ_diff_m  = dim_avg_n_Wrap(mdQ_diff_zm,0)

  cdQ_m = dim_avg_n_Wrap(cdQ_zm,0)
  cdQ_ml_m  = dim_avg_n_Wrap(cdQ_ml_zm,0)
  cdQ_diff_m  = dim_avg_n_Wrap(cdQ_diff_zm,0)

; check range
printMinMax(mdQ_m,True)
printMinMax(mdQ_ml_m,True)
printMinMax(mdQ_diff_m,True)

printMinMax(cdQ_m,True)
printMinMax(cdQ_ml_m,True)
printMinMax(cdQ_diff_m,True)


; initialize panel plot
  wks  = gsn_open_wks(pltType,name)
  gsn_define_colormap(wks,"WhViBlGrYeOrReWh")

plot = new(6,graphic)

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
;        pres@gsnPanelMainString       = "Moist dQ/dt Field"  ; Adjust name!
        pres@gsnPanelMainFontHeightF  = 0.015

	; res1@cnLevelSelectionMode = "ExplicitLevels" 
	res1@cnLevelSelectionMode = "ManualLevels" 

	res1@gsnRightString = "g/kg/day"
	res1@tiYAxisString   = "Pressure (hPa)"

	res1@gsnCenterString = "CAM"

	; res1@cnLevels =       (/ -3.0,   -3.0,   -2.0,   -1.0,   -0.5,   0.0,   0.5,   1.0,   2.0,   2.3/)
	; res1@lbLabelStrings = (/"-3.0", "-3.0", "-2.0", "-1.0", "-0.5", "0.0", "0.5", "1.0", "2.0", "2.3"/)
        res1@cnMaxLevelValF  =    2.5                 ; max contour color label
        res1@cnMinLevelValF  =   -2.5                  ; min contour color label
        res1@cnLevelSpacingF = 0.5                         ; contour color spacing

	res1@gsnCenterString = "CAM"
	res1@gsnLeftString = "Moist dq/dt"
        plot(0) = gsn_csm_contour(wks,mdQ_m(:,:),res1)	; plot var

	res1@gsnCenterString = "ML"
	plot(1) = gsn_csm_contour(wks,mdQ_ml_m(:,:),res1)	; plot var

	res1@gsnCenterString = "ML - CAM"

	; res1@cnLevels =       (/ -0.05,   -0.03,   -0.01,   0.0,   0.01,   0.03,   0.05,   0.08,   0.1,   1.2/)
	; res1@lbLabelStrings = (/"-0.05", "-0.03", "-0.01", "0.0", "0.01", "0.03", "0.05", "0.08", "0.1", "1.2"/)
        res1@cnMaxLevelValF  =    0.05                 ; max contour color label
        res1@cnMinLevelValF  =   -0.05                 ; min contour color label
        res1@cnLevelSpacingF = 0.01                           ; contour color spacing

	plot(2) = gsn_csm_contour(wks,mdQ_diff_m(:,:),res1)	; plot var

;;;;;;;


	; res1@cnLevels =       (/ -1.6,   -1.0,   -0.4,   -0.1,   0.0,   0.1,   0.4,   1.0,   1.4,   1.75/)
	; res1@lbLabelStrings = (/"-1.6", "-1.0", "-0.4", "-0.1", "0.0", "0.1", "0.4", "1.0", "1.4", "1.75"/)
        res1@cnMaxLevelValF  =   1.5                 ; max contour color label
        res1@cnMinLevelValF  =   -1.5                  ; min contour color label
        res1@cnLevelSpacingF = 0.25                         ; contour color spacing
	res1@gsnCenterString = "CAM"
	res1@gsnLeftString = "Conv. dq/dt"
        plot(3) = gsn_csm_contour(wks,cdQ_m(:,:),res1)	; plot var

	res1@gsnCenterString = "ML"
	plot(4) = gsn_csm_contour(wks,cdQ_ml_m(:,:),res1)	; plot var

	res1@gsnCenterString = "ML - CAM"

	; res1@cnLevels =       (/ -0.04,   -0.03,   -0.02,   -0.01,   -0.05,   0.0,   0.05,   0.01,   0.02,   0.03/)
	; res1@lbLabelStrings = (/"-0.04", "-0.03", "-0.02", "-0.01", "-0.05", "0.0", "0.05", "0.01", "0.02", "0.03"/)
        res1@cnMaxLevelValF  =    0.04                 ; max contour color label
        res1@cnMinLevelValF  =   -0.04                 ; min contour color label
        res1@cnLevelSpacingF = 0.01                           ; contour color spacing

	plot(5) = gsn_csm_contour(wks,cdQ_diff_m(:,:),res1)	; plot var


	gsn_panel(wks,plot,panelSize,pres)			; panel plot
	frame(wks)
end
