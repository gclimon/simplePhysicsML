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
  fname = "results_50trees.nc"
  
  name = "dQdt_snap"

  mdT_f = addfile(moist + "PTEQ/rf/results_aug.nc","r")
  cdT_f = addfile(conv + "PTEQ/rf/"+fname,"r")

; output format
;  pltType = "X11"
;  pltType = "png"
  pltType = "eps"
  panelSize = (/2,3/)    ; panel dimensions
  version   = ""

  lat  = mdT_f->lat
  lon  = mdT_f->lon
  pi    = 4.*atan(1.)
  lev  = mdT_f->lev

  t = 25

; choose the variable
  mdT = mdT_f->PTEQ(t,23,:,:)
  mdT_ml = mdT_f->PTEQ_ML_predicted(t,23,:,:)
  mdT_diff = mdT_f->PTEQ_difference(t,23,:,:)
  cdT = cdT_f->PTEQ(t,23,:,:)
  cdT_ml = cdT_f->PTEQ_ML_predicted(t,23,:,:)
  cdT_diff = cdT_f->PTEQ_difference(t,23,:,:)


  mdT_m = mdT * 86400. * 1000.
  copy_VarMeta(mdT,mdT_m)
  mdT_ml_m = mdT_ml * 86400. * 1000.
  copy_VarMeta(mdT_ml,mdT_ml_m)
  mdT_diff_m = mdT_diff * 86400. * 1000.  ; ML-CAM
  copy_VarMeta(mdT_diff,mdT_diff_m)

  cdT_m = cdT * 86400. * 1000.
  copy_VarMeta(cdT,cdT_m)
  cdT_ml_m = cdT_ml * 86400. * 1000.
  copy_VarMeta(cdT_ml,cdT_ml_m)
  cdT_diff_m = cdT_diff * 86400. * 1000.  ; ML-CAM
  copy_VarMeta(cdT_diff,cdT_diff_m)

printMinMax(mdT_m,True)
printMinMax(mdT_ml_m,True)
printMinMax(mdT_diff_m,True)

printMinMax(cdT_m,True)
printMinMax(cdT_ml_m,True)
printMinMax(cdT_diff_m,True)

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
	res1@vpHeightF            = 0.25                        ; set aspect ratio
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

	res1@gsnRightString = "g/kg/day"
;	res1@tiYAxisString   = "Pressure (hPa)"

;;;;;;;

	; res1@cnLevels = (/-2.0, -1.0, 0.0, 1.5, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0/)	
	; res1@lbLabelStrings = (/"-2.0", "-1.0", "0.0", "1.5", "3.0", "4.0", "5.0", "6.0", "7.0", "8.0"/)	
        res1@cnMaxLevelValF  =   5.                 ; max contour color label
        res1@cnMinLevelValF  =   -5.                  ; min contour color label
        res1@cnLevelSpacingF =   1.                       ; contour color spacing

	res1@gsnCenterString = "CAM"
	res1@gsnLeftString = "Moist dq/dt"
        plot(0) = gsn_csm_contour(wks,mdT_m(:,:),res1)	; plot var

	res1@gsnCenterString = "ML"
	plot(1) = gsn_csm_contour(wks,mdT_ml_m(:,:),res1)	; plot var

	res1@gsnCenterString = "ML - CAM"

	; res1@cnLevels = (/-1.3, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.00, 2.6, 3.3/)	
	; res1@lbLabelStrings = (/"-1.3", "-1.0", "-0.5", "0.0", "0.5", "1.0", "1.5", "2.0", "2.6", "3.3"/)		
        res1@cnMaxLevelValF  =    1.0                 ; max contour color label
        res1@cnMinLevelValF  =   -1.0                 ; min contour color label
        res1@cnLevelSpacingF = 0.25                           ; contour color spacing

	plot(2) = gsn_csm_contour(wks,mdT_diff_m(:,:),res1)	; plot var

;;;;;;;

	; res1@cnLevels = (/-9.0, -7.0, -5.0, -3.0, -1.0, 0.0, 1.0, 3.0, 5.0, 7.0/)
	; res1@lbLabelStrings = (/"-9.0", "-7.0", "-5.0", "-3.0", "1.0", "0.0", "1.0", "3.0", "5.0", "7.0"/)
	res1@gsnCenterString = "CAM"
       	res1@gsnLeftString = "Conv. dq/dt"

        res1@cnMaxLevelValF  =    10.0                 ; max contour color label
        res1@cnMinLevelValF  =   -10.0                 ; min contour color label
        res1@cnLevelSpacingF = 2.0                           ; contour color spacing

	plot(3) = gsn_csm_contour(wks,cdT_m(:,:),res1)	; plot var

	; res1@cnLevels = (/-2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0, 2.3/)
	; res1@lbLabelStrings = (/"-2.0", "-1.5", "-1.0", "-0.5", "0.0", "0.5", "1.0", "1.5", "2.0", "2.3"/)


	res1@gsnCenterString = "ML"
	plot(4) = gsn_csm_contour(wks,cdT_ml_m(:,:),res1)	; plot var

        res1@cnMaxLevelValF  =    1.0                 ; max contour color label
        res1@cnMinLevelValF  =   -1.0                 ; min contour color label
        res1@cnLevelSpacingF =    0.25                           ; contour color spacing

	res1@gsnCenterString = "ML - CAM"
	plot(5) = gsn_csm_contour(wks,cdT_diff_m(:,:),res1)	; plot var

	gsn_panel(wks,plot,panelSize,pres)			; panel plot
	frame(wks)

end
