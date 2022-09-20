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


  path = "/glade/scratch/glimon/ml_climate/19L/"
  moist = path+"moist/"
  wRH  = addfile(moist + "PTEQ/rf/R2_wRH.nc" ,"r")
  noRH = addfile(moist + "PTEQ/rf/R2_noRH.nc","r")

  name = "R2_RH"

; output format
;  pltType = "X11"
;  pltType = "png"
  pltType = "eps"
  panelSize = (/1,2/)    ; panel dimensions
  version   = ""

  lat  = wRH->lat
  lon  = wRH->lon
  pi    = 4.*atan(1.)
  lev  = wRH->lev
; read the surface pressure

; choose the variable

  wRHdq = wRH->R2(:,:)
  noRHdq = noRH->R2(:,:)

  wks  = gsn_open_wks(pltType,name+version)
  gsn_define_colormap(wks,"WhViBlGrYeOrRe")

plot = new(2,graphic)

	res1                      = True
	res1@gsnDraw              = False          		; panel plot 
	res1@gsnFrame             = False			; don't draw yet
	res1@cnFillOn             = True
	res1@cnLinesOn            = True
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
        pres@gsnPanelFigureStrings    = (/"a)","b)","c)","d)","e)","f)","g)","g)"/) ; add strings to panel
        pres@gsnPanelFigureStringsFontHeightF = 0.01
        pres@amJust                   = "TopLeft"
        pres@gsnAttachBorderOn        = "False"
;        pres@gsnPanelMainString       = ""  ; Adjust name!
        pres@gsnPanelMainFontHeightF  = 0.015

	res1@cnLevelSelectionMode = "ExplicitLevels" 
        res1@cnLevels = (/0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99/)
	res1@lbLabelStrings = (/"0.0", "0.1", "0.2", "0.3", "0.4", "0.5", "0.6", "0.7", "0.8", "0.9", "0.95", "0.99"/)
	res1@lbOrientation = "vertical"
        ; res1@cnMaxLevelValF  =   0.99                 ; max contour color label
        ; res1@cnMinLevelValF  =   0.0                  ; min contour color label
        ; res1@cnLevelSpacingF = 0.1                       ; contour color spacing

	res1@tiYAxisString   = "Pressure (hPa)"	

	res1@gsnCenterString = "R2 (w/ RH)"
	res1@gsnRightString = "dq/dt"
	res1@gsnLeftString = "Moist"
	plot(0) = gsn_csm_contour(wks,wRHdq(:,:),res1)	; plot var

	res1@gsnCenterString = "R2 (no RH)"
	res1@gsnLeftString = "Moist"
	plot(1) = gsn_csm_contour(wks,noRHdq(:,:),res1)	; plot var

	gsn_panel(wks,plot,panelSize,pres)			; panel plot
	frame(wks)
end