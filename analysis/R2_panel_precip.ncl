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

  fname = "R2_50trees.nc"
  path = "/glade/scratch/glimon/simplePhysML/19/"
  moist = path+"TJ/"
  conv = path+"TJBM/"
  ; ddT = addfile(dry + "PTTEND/rf/"+fname,"r")
  ; mdT = addfile(moist + "PTTEND/rf/"+fname,"r")
  ; cdT = addfile(conv + "PTTEND/rf/"+fname,"r")
  ; mdQ = addfile(moist + "PTEQ/rf/"+fname,"r")
  ; cdQ = addfile(conv + "PTEQ/rf/"+fname,"r")
  mPL = addfile(moist + "PRECL/rf/R2_aug.nc","r")
  cPL = addfile(conv + "PRECL/rf/"+fname,"r")
  cPC = addfile(conv + "PRECC/rf/"+fname,"r")

  name = "R2_panel_precip"

; output format
;  pltType = "X11"
;  pltType = "png"
  pltType = "eps"
  panelSize = (/1,1/)    ; panel dimensions
  version   = ""

  lat  = mPL->lat
  lon  = mPL->lon
  pi    = 4.*atan(1.)
;  lev  = mdT->lev
; read the surface pressure

; choose the variable

  R2mPL = mPL->R2(:)
  R2cPL = cPL->R2(:)
  R2cPC = cPC->R2(:)

  R2cP = new((/3,dimsizes(R2cPL)/),double)
  R2cP(0,:) = R2mPL
  R2cP(1,:) = R2cPL
  R2cP(2,:) = R2cPC

  wks  = gsn_open_wks(pltType,name+version)
  gsn_define_colormap(wks,"WhViBlGrYeOrRe")

plot = new(1,graphic)

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
        pres@gsnPanelFigureStrings    = (/"a)","b)","c)","c)","e)","f)","g)","g)"/) ; add strings to panel
        pres@gsnPanelFigureStringsFontHeightF = 0.01
        pres@amJust                   = "TopLeft"
        pres@gsnAttachBorderOn        = "False"
;        pres@gsnPanelMainString       = ""  ; Adjust name!
        pres@gsnPanelMainFontHeightF  = 0.015

	res1@xyLineThicknesses = (/  1.0, 1.0, 1.0/)          ; make second line thicker
	res1@xyLineColors      = (/"orange","blue","black"/)          ; change line color

	res1@cnLevelSelectionMode = "ExplicitLevels" 
        res1@cnLevels = (/0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99/)
	res1@lbLabelStrings = (/"0.0", "0.1", "0.2", "0.3", "0.4", "0.5", "0.6", "0.7", "0.8", "0.9", "0.95", "0.99"/)
	res1@lbOrientation = "vertical"
        ; res1@cnMaxLevelValF  =   0.99                 ; max contour color label
        ; res1@cnMinLevelValF  =   0.0                  ; min contour color label
        ; res1@cnLevelSpacingF = 0.1                       ; contour color spacing

	; res1@vpWidthF             = 0.5                         ; set aspect ratio
	; res1@vpHeightF            = 0.25                        ; set aspect ratio

	res1@pmLegendDisplayMode    = "Always"            ; turn on legend
  	res1@pmLegendSide           = "Top"               ; Change location of 
  	res1@pmLegendParallelPosF   = .15                  ; move units ri
	res1@pmLegendOrthogonalPosF = -1.                ; move units down
  	res1@pmLegendWidthF         = 0.05                ; Change width and
  	res1@pmLegendHeightF        = 0.08                ; height of legend.
  	res1@lgPerimOn              = False               ; turn off box around
  	res1@lgLabelFontHeightF     = .01                 ; label font height
  	res1@xyExplicitLegendLabels = (/"Moist Large Scale","Convection Large Scale","Convection Convective"/)         ; create explicit labels

        res1@gsnPanelMainFontHeightF  = 0.01
	res1@trYReverse           = False      			; reverses y-axis, fals
	res1@trYMaxF = 1.00
;	res1@trYMinF = 0.80
	res1@trYMinF = 0.87
	res1@gsnRightString = ""
	res1@tiYAxisString   = "R2"
	res1@gsnLeftString = ""
	res1@gsnCenterString = "Precipitation"
;	plot(0) = gsn_csm_xy(wks,lat,R2mPL(:),res1)	; plot var

;	res1@gsnLeftString = "Convection"
;	res1@gsnCenterString = "Convection Precip"
;	plot(0) = gsn_csm_xy(wks,lat,R2cPL(:),res1)	; plot var

;	res1@gsnCenterString = "Convectve Precip"
	plot(0) = gsn_csm_xy(wks,lat,R2cP,res1)	; plot var

	gsn_panel(wks,plot,panelSize,pres)			; panel plot
	frame(wks)
end