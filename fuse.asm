//(c) 2026, Retired Coder(RC)

//submod+nulmod+copy fused
FUNCTION CalcToInv_FusedA()
{
    [B------:R-:W-:-:S04]    IADD3.X RFirst0, Pt0, PT, RFirst0, ~RSub0, RZ, !PT, PT
    [B------:R-:W-:-:S04]    IADD3.X RFirst1, Pt0, PT, RFirst1, ~RSub1, RZ, Pt0, !PT
    [B------:R-:W-:-:S04]    IADD3.X RFirst2, Pt0, PT, RFirst2, ~RSub2, RZ, Pt0, !PT
    [B------:R-:W-:-:S04]    IADD3.X RFirst3, Pt0, PT, RFirst3, ~RSub3, RZ, Pt0, !PT
    [B------:R-:W-:-:S04]    IADD3.X RFirst4, Pt0, PT, RFirst4, ~RSub4, RZ, Pt0, !PT
    [B------:R-:W-:-:S04]    IADD3.X RFirst5, Pt0, PT, RFirst5, ~RSub5, RZ, Pt0, !PT
    [B------:R-:W-:-:S04]    IADD3.X RFirst6, Pt0, PT, RFirst6, ~RSub6, RZ, Pt0, !PT
    [B------:R-:W-:Y:S13]    IADD3.X RFirst7, Pt0, PT, RFirst7, ~RSub7, RZ, Pt0, !PT
//add P if <0
    [B------:R-:W-:-:S04] @!Pt0 IADD3.X RFirst0, Pt1, PT, RFirst0, 0xFFFFFC2F, RZ, !PT, !PT
    [B------:R-:W-:-:S04] @!Pt0 IADD3.X RFirst1, Pt1, PT, RFirst1, 0xFFFFFFFE, RZ, Pt1, !PT
    [B------:R-:W-:-:S02]    IMAD.WIDE.U32 Ro0, RFirst0.reuse, RSecond0, RZ
    [B------:R-:W-:-:S02]    IMAD.WIDE.U32 Ro2, RFirst0, RSecond2, RZ
//2: 1B 0E
    [B------:R-:W-:-:S02]    IMAD.WIDE.U32 Ro2, Pt4, RFirst1.reuse, RSecond1, Ro2
    [B------:R-:W-:-:S02] @!Pt0 IADD3.X RFirst2, Pt1, PT, RFirst2, 0xFFFFFFFF, RZ, Pt1, !PT
    [B------:R-:W-:-:S02]    IMAD.WIDE.U32.X Ro4, RFirst0, RSecond4, RZ, Pt4
//2: 2A 1D 0G
    [B------:R-:W-:-:S02]    IMAD.WIDE.U32 Ro2, Pt4, RFirst2, RSecond0, Ro2
    [B------:R-:W-:-:S02] @!Pt0 IADD3.X RFirst3, Pt1, PT, RFirst3, 0xFFFFFFFF, RZ, Pt1, !PT
    [B------:R-:W-:-:S02]    IMAD.WIDE.U32.X Ro4, Pt4, RFirst1.reuse, RSecond3, Ro4, Pt4
    [B------:R-:W-:-:S02] @!Pt0 IADD3.X RFirst4, Pt1, PT, RFirst4, 0xFFFFFFFF, RZ, Pt1, !PT
    [B------:R-:W-:-:S02]    IMAD.WIDE.U32.X Ro6, RFirst0, RSecond6, RZ, Pt4
//4: 2C 1F 1H
    [B------:R-:W-:-:S02]    IMAD.WIDE.U32 Ro4, Pt4, RFirst2, RSecond2, Ro4
    [B------:R-:W-:-:S02] @!Pt0 IADD3.X RFirst5, Pt1, PT, RFirst5, 0xFFFFFFFF, RZ, Pt1, !PT
    [B------:R-:W-:-:S02]    IMAD.WIDE.U32.X Ro6, Pt4, RFirst1.reuse, RSecond5, Ro6, Pt4
    [B------:R-:W-:-:S02] @!Pt0 IADD3.X RFirst6, Pt1, PT, RFirst6, 0xFFFFFFFF, RZ, Pt1, !PT
    [B------:R-:W-:-:S02]    IMAD.WIDE.U32.X Rt12, RFirst1, RSecond7, RZ, Pt4
//4: 3B 2E 2G 3H
    [B------:R-:W-:-:S02]    IMAD.WIDE.U32 Ro4, Pt4, RFirst3, RSecond1, Ro4
    [B------:R-:W-:-:S02] @!Pt0 IADD3.X RFirst7, PT, PT, RFirst7, 0xFFFFFFFF, RZ, Pt1, !PT
    [B------:R-:W-:Y:S04]    IMAD.WIDE.U32.X Ro6, Pt4, RFirst2.reuse, RSecond4, Ro6, Pt4
    [B------:R-:W-:Y:S04]    IMAD.WIDE.U32.X Rt12, Pt4, RFirst2, RSecond6, Rt12, Pt4
    [B------:R-:W-:-:S02]    IMAD.WIDE.U32.X Rt14, PT, RFirst3.reuse, RSecond7, RZ, Pt4
//4: 4A 3D 3F 4G 5H
    [B------:R-:W-:Y:S04]    IMAD.WIDE.U32 Ro4, Pt4, RFirst4, RSecond0, Ro4
    [B------:R-:W-:Y:S04]    IMAD.WIDE.U32.X Ro6, Pt4, RFirst3.reuse, RSecond3, Ro6, Pt4
    [B------:R-:W-:Y:S04]    IMAD.WIDE.U32.X Rt12, Pt4, RFirst3, RSecond5, Rt12, Pt4
    [B------:R-:W-:Y:S04]    IMAD.WIDE.U32.X Rt14, Pt4, RFirst4.reuse, RSecond6, Rt14, Pt4
    [B------:R-:W-:-:S02]    IMAD.WIDE.U32.X Rt16, RFirst5, RSecond7, RZ, Pt4
//6: 4C 4E 5F 6G 7H
    [B------:R-:W-:Y:S04]    IMAD.WIDE.U32 Ro6, Pt4, RFirst4.reuse, RSecond2, Ro6
    [B------:R-:W-:Y:S04]    IMAD.WIDE.U32.X Rt12, Pt4, RFirst4, RSecond4, Rt12, Pt4
    [B------:R-:W-:Y:S04]    IMAD.WIDE.U32.X Rt14, Pt4, RFirst5, RSecond5, Rt14, Pt4
    [B------:R-:W-:Y:S04]    IMAD.WIDE.U32.X Rt16, Pt4, RFirst6, RSecond6, Rt16, Pt4
    [B------:R-:W-:-:S02]    IMAD.WIDE.U32.X Rt18, RFirst7, RSecond7, RZ, Pt4
//6: 5B 5D 6E 7F, get C3
    [B------:R-:W-:Y:S04]    IMAD.WIDE.U32 Ro6, Pt4, RFirst5.reuse, RSecond1, Ro6
    [B------:R-:W-:Y:S04]    IMAD.WIDE.U32.X Rt12, Pt4, RFirst5, RSecond3, Rt12, Pt4
    [B------:R-:W-:Y:S04]    IMAD.WIDE.U32.X Rt14, Pt4, RFirst6.reuse, RSecond4, Rt14, Pt4
    [B------:R-:W-:-:S02]    IMAD.WIDE.U32.X Rt16, Pt3, RFirst7, RSecond5, Rt16, Pt4
//6: 6A 6C 7D, get C2
    [B------:R-:W-:Y:S04]    IMAD.WIDE.U32 Ro6, Pt4, RFirst6.reuse, RSecond0, Ro6
    [B------:R-:W-:Y:S04]    IMAD.WIDE.U32.X Rt12, Pt4, RFirst6, RSecond2, Rt12, Pt4
    [B------:R-:W-:-:S02]    IMAD.WIDE.U32.X Rt14, Pt2, RFirst7.reuse, RSecond3, Rt14, Pt4
//8: 7B, get C1
    [B------:R-:W-:-:S02]    IMAD.WIDE.U32 Rt12, Pt1, RFirst7, RSecond1, Rt12
///////////////////////////////
//0: 0B
    [B------:R-:W-:-:S01]    IMAD.WIDE.U32 Rt0, RFirst0, RSecond1, RZ
    [B------:R-:W-:-:S01]    IADD3.X Rt10, RZ, RZ, RZ, Pt1, !PT //C1 to Rt10
//0: 1A 0D
    [B------:R-:W-:Y:S04]    IMAD.WIDE.U32 Rt0, Pt0, RFirst1, RSecond0, Rt0
    [B------:R-:W-:-:S01]    IMAD.WIDE.U32.X Rt2, RFirst0, RSecond3, RZ, Pt0
    [B------:R-:W-:-:S01]    IADD3.X Rt11, RZ, RZ, RZ, Pt2, !PT //C2 to Rt11
//2: 1C 0F
    [B------:R-:W-:-:S03]    IMAD.WIDE.U32 Rt2, Pt0, RFirst1, RSecond2, Rt2
    [B------:R-:W-:-:S01]    IADD3.X Rt18, Pt1, PT, Rt18, RZ, RZ, Pt3, !PT //now use C3 and C7 and forget
    [B------:R-:W-:-:S01]    IMAD.WIDE.U32.X Rt4, RFirst0, RSecond5, RZ, Pt0
    [B------:R-:W-:-:S01]    IADD3 Ro1, Pt3, Ro1, Rt0, RZ //build res
//2: 2B 1E 0H
    [B------:R-:W-:-:S03]    IMAD.WIDE.U32 Rt2, Pt0, RFirst2, RSecond1, Rt2
    [B------:R-:W-:-:S01]    IADD3.X Ro2, Pt3, Ro2, Rt1, RZ, Pt3, !PT //build res
    [B------:R-:W-:-:S04]    IMAD.WIDE.U32.X Rt4, Pt0, RFirst1, RSecond4, Rt4, Pt0
    [B------:R-:W-:-:S01]    IMAD.WIDE.U32.X Rt0, RFirst0, RSecond7, RZ, Pt0
//2: 3A 2D 1G 2H
    [B------:R-:W-:Y:S04]    IMAD.WIDE.U32 Rt2, Pt0, RFirst3, RSecond0, Rt2
    [B------:R-:W-:-:S02]    IMAD.WIDE.U32.X Rt4, Pt0, RFirst2, RSecond3, Rt4, Pt0
    [B------:R-:W-:-:S01]    IADD3.X Ro3, Pt3, Ro3, Rt2, RZ, Pt3, !PT //build res
    [B------:R-:W-:-:S03]    IMAD.WIDE.U32.X Rt0, Pt0, RFirst1, RSecond6, Rt0, Pt0
    [B------:R-:W-:-:S01]    IADD3.X Ro4, Pt3, Ro4, Rt3, RZ, Pt3, !PT //build res
    [B------:R-:W-:-:S01]    IMAD.WIDE.U32.X Rt2, RFirst2, RSecond7, RZ, Pt0
//4: 3C 2F 3G 4H
    [B------:R-:W-:Y:S04]    IMAD.WIDE.U32 Rt4, Pt0, RFirst3, RSecond2, Rt4
    [B------:R-:W-:Y:S04]    IMAD.WIDE.U32.X Rt0, Pt0, RFirst2, RSecond5, Rt0, Pt0
    [B------:R-:W-:Y:S04]    IMAD.WIDE.U32.X Rt2, Pt0, RFirst3, RSecond6, Rt2, Pt0
    [B------:R-:W-:-:S01]    IMAD.WIDE.U32.X Rt6, RFirst4, RSecond7, RZ, Pt0
//4: 4B 3E 4F 5G 6H
    [B------:R-:W-:Y:S04]    IMAD.WIDE.U32 Rt4, Pt0, RFirst4, RSecond1, Rt4
    [B------:R-:W-:Y:S04]    IMAD.WIDE.U32.X Rt0, Pt0, RFirst3, RSecond4, Rt0, Pt0
    [B------:R-:W-:Y:S04]    IMAD.WIDE.U32.X Rt2, Pt0, RFirst4, RSecond5, Rt2, Pt0
    [B------:R-:W-:Y:S04]    IMAD.WIDE.U32.X Rt6, Pt0, RFirst5, RSecond6, Rt6, Pt0
    [B------:R-:W-:-:S01]    IMAD.WIDE.U32.X Rt8, RFirst6, RSecond7, RZ, Pt0
//4: 5A 4D 5E 6F 7G, get C7(store in Pt4)
    [B------:R-:W-:Y:S04]    IMAD.WIDE.U32 Rt4, Pt0, RFirst5, RSecond0, Rt4
    [B------:R-:W-:-:S02]    IMAD.WIDE.U32.X Rt0, Pt0, RFirst4, RSecond3, Rt0, Pt0
    [B------:R-:W-:-:S01]    IADD3.X Ro5, Pt3, Ro5, Rt4, RZ, Pt3, !PT //build res
    [B------:R-:W-:-:S02]    IMAD.WIDE.U32.X Rt2, Pt0, RFirst5, RSecond4, Rt2, Pt0
    [B------:R-:W-:-:S01]    IADD3.X Ro6, Pt3, Ro6, Rt5, RZ, Pt3, !PT //build res
    [B------:R-:W-:Y:S04]    IMAD.WIDE.U32.X Rt6, Pt0, RFirst6, RSecond5, Rt6, Pt0
    [B------:R-:W-:-:S01]    IMAD.WIDE.U32.X Rt8, Pt4, RFirst7, RSecond6, Rt8, Pt0
//6: 5C 6D 7E, get C6
    [B------:R-:W-:-:S03]    IMAD.WIDE.U32 Rt0, Pt0, RFirst5, RSecond2, Rt0
    [B------:R-:W-:-:S01]    IADD3.X Rt19, PT, PT, Rt19, RZ, RZ, Pt1, Pt4 //now use C3 and C7 and forget
    [B------:R-:W-:Y:S04]    IMAD.WIDE.U32.X Rt2, Pt0, RFirst6, RSecond3, Rt2, Pt0
    [B------:R-:W-:-:S01]    IMAD.WIDE.U32.X Rt6, Pt1, RFirst7, RSecond4, Rt6, Pt0 //C6 to Pt1
//6: 6B 7C, get C5
    [B------:R-:W-:Y:S04]    IMAD.WIDE.U32 Rt0, Pt0, RFirst6, RSecond1, Rt0

    [B------:R-:W-:-:S01]    IMAD.WIDE.U32.X Rt2, Pt2, RFirst7, RSecond2, Rt2, Pt0
//6: 7A, get C4
    [B------:R-:W-:-:S01]    IMAD.WIDE.U32 Rt0, Pt4, RFirst7, RSecond0, Rt0
    [B------:R-:W-:-:S01]    IADD3.X Rt4, RZ, RZ, RZ, Pt2, !PT //C5 to Rt4
    [B------:R-:W-:Y:S04]    IADD3.X Ro7, Pt3, Ro7, Rt0, RZ, Pt3, !PT //build res
    [B------:R-:W-:Y:S04]    IADD3.X Rt12, Pt3, Rt12, Rt1, RZ, Pt3, !PT //build res
    [B------:R-:W-:-:S01]    IADD3.X Rt13, Pt3, Pt4, Rt13, Rt2, RZ, Pt3, Pt4 //build res
    [B------:R-:W-:-:S01]    IMAD.WIDE.U32 Rt0, Rt12, 0x3D1, RZ
    [B------:R-:W-:-:S01]    IADD3.X Rt14, Pt3, Pt4, Rt14, Rt3, Rt10, Pt3, Pt4 //build res
    [B------:R-:W-:-:S01]    IADD3.X Rt10, RZ, RZ, RZ, Pt1, !PT //C6 to Rt10
    [B------:R-:W-:-:S01]    IMAD.WIDE.U32.X Rt2, Pt2, Rt13, 0x3D1, Rt12, !PT
    [B------:R-:W-:-:S01]    IADD3.X Rt15, Pt3, Pt4, Rt15, Rt6, Rt4, Pt3, Pt4 //build res
    [B------:R-:W-:-:S01]    IADD3.X Ro0, Pt0, Pt1, Ro0, Rt0, RZ, !PT, !PT
    [B------:R-:W-:-:S01]    IADD3.X Rt16, Pt3, Pt4, Rt16, Rt7, Rt11, Pt3, Pt4 //build res
    [B------:R-:W-:-:S01]    IMAD.WIDE.U32 Rt4, Rt14, 0x3D1, RZ
    [B------:R-:W-:-:S01]    IADD3.X Rt17, Pt3, Pt4, Rt17, Rt8, Rt10, Pt3, Pt4 //build res
    [B------:R-:W-:-:S01]    IADD3.X Ro1, Pt0, Pt1, Rt1, Rt2, Ro1, Pt0, Pt1
    [B------:R-:W-:-:S01]    IADD3.X Rt18, Pt3, Pt4, Rt18, Rt9, RZ, Pt3, Pt4 //build res
    [B------:R-:W-:-:S01]    IMAD.WIDE.U32.X Rt6, Pt2, Rt15, 0x3D1, Rt14, Pt2
    [B------:R-:W-:-:S01]    IADD3.X Rt19, Rt19, RZ, RZ, Pt3, Pt4 //build res
// reduction
    [B------:R-:W-:-:S01]    IADD3.X Ro2, Pt0, Pt1, Rt3, Rt4, Ro2, Pt0, Pt1
    [B------:R-:W-:-:S01]    IMAD.WIDE.U32 Rt8, Rt16, 0x3D1, RZ
    [B------:R-:W-:-:S01]    IADD3.X Ro3, Pt0, Pt1, Rt5, Rt6, Ro3, Pt0, Pt1
    [B------:R-:W-:-:S01]    IMAD.WIDE.U32.X Rt4, Pt2, Rt17, 0x3D1, Rt16, Pt2
    [B------:R-:W-:-:S01]    IMAD.WIDE.U32 Rt0, Rt18, 0x3D1, RZ
    [B------:R-:W-:-:S01]    IADD3.X Ro4, Pt0, Pt1, Rt7, Rt8, Ro4, Pt0, Pt1
    [B------:R-:W-:-:S01]    IMAD.WIDE.U32.X Rt2, Pt2, Rt19, 0x3D1, Rt18, Pt2
    [B------:R-:W-:Y:S04]    IADD3.X Ro5, Pt0, Pt1, Rt9, Rt4, Ro5, Pt0, Pt1
    [B------:R-:W-:Y:S04]    IADD3.X Ro6, Pt0, Pt1, Rt5, Rt0, Ro6, Pt0, Pt1
    [B------:R-:W-:Y:S04]    IADD3.X Ro7, Pt0, Pt1, Rt1, Rt2, Ro7, Pt0, Pt1
//Rt8..Rt9 is "tmp[4] in cpu code, Rt8 <= 0x3d1 (checked P-1*P-1), Rt9 <= 1
    [B------:R-:W-:Y:S04]    IADD3.X Rt8, Pt0, PT, Rt3, RZ, RZ, Pt0, Pt1
    [B------:R-:W-:Y:S01]    IADD3.X Rt9, Pt0, Pt1, RZ, RZ, RZ, Pt0, Pt2
////
    [B------:R-:W-:-:S01]    IMAD.WIDE.U32.X Ro0, Pt3, Rt8, 0x3D1, Ro0, !PT
    [B------:R-:W-:-:S01]    IMAD.WIDE.U32 Rt2, Rt9, 0x3D1, Rt8 //no carry occurs!
    [B------:R-:W-:-:S01]    MOV RCopy0, Ro0
    [B------:R-:W-:Y:S04]    IADD3.X Ro1, Pt0, PT, Ro1, Rt2, RZ, !PT, !PT
    [B------:R-:W-:-:S03]    IADD3.X Ro2, Pt0, Pt1, Rt3, Ro2, RZ, Pt0, Pt3
    [B------:R-:W-:-:S01]    MOV RCopy1, Ro1
    [B------:R-:W-:-:S03]    IADD3.X Ro3, Pt0, PT, Ro3, RZ, RZ, Pt0, Pt1
    [B------:R-:W-:-:S01]    MOV RCopy2, Ro2
    [B------:R-:W-:-:S01]    IADD3.X Ro4, Pt0, PT, Ro4, RZ, RZ, Pt0, !PT
    [B------:R-:W-:-:S01]    MOV RCopy3, Ro3
    [B------:R-:W-:-:S03]    IADD3.X Ro5, Pt0, PT, Ro5, RZ, RZ, Pt0, !PT
    [B------:R-:W-:-:S01]    MOV RCopy4, Ro4
    [B------:R-:W-:-:S03]    IADD3.X Ro6, Pt0, PT, Ro6, RZ, RZ, Pt0, !PT
    [B------:R-:W-:-:S01]    MOV RCopy5, Ro5
    [B------:R-:W-:-:S03]    IADD3.X Ro7, PT, PT, Ro7, RZ, RZ, Pt0, !PT
    [B------:R-:W-:-:S01]    IMAD RCopy6, RZ, RZ, Ro6
    [B------:R-:W-:-:S05]    MOV RCopy7, Ro7
}


FUNCTION CalcNewPoints_FusedA()
{
    [B------:R-:W-:-:S02]    IMAD.WIDE.U32 Ro0, RFirst0.reuse, RSecond0, RZ
    [B------:R-:W-:-:S02]    IADD3.X RSubOut0, Pt0, PT, RSubFirst0, ~RSubSecond0, RZ, !PT, PT
    [B------:R-:W-:-:S02]    IMAD.WIDE.U32 Ro2, RFirst0, RSecond2, RZ
    [B------:R-:W-:-:S02]    IADD3.X RSubOut1, Pt0, PT, RSubFirst1, ~RSubSecond1, RZ, Pt0, !PT
    [B------:R-:W-:-:S02]    IMAD.WIDE.U32 Ro2, Pt4, RFirst1.reuse, RSecond1, Ro2
    [B------:R-:W-:-:S02]    IADD3.X RSubOut2, Pt0, PT, RSubFirst2, ~RSubSecond2, RZ, Pt0, !PT
    [B------:R-:W-:-:S02]    IMAD.WIDE.U32.X Ro4, RFirst0, RSecond4, RZ, Pt4
    [B------:R-:W-:-:S02]    IMAD.WIDE.U32 Ro2, Pt4, RFirst2, RSecond0, Ro2
    [B------:R-:W-:-:S02]    IADD3.X RSubOut3, Pt0, PT, RSubFirst3, ~RSubSecond3, RZ, Pt0, !PT
    [B------:R-:W-:-:S02]    IMAD.WIDE.U32.X Ro4, Pt4, RFirst1.reuse, RSecond3, Ro4, Pt4
    [B------:R-:W-:-:S02]    IADD3.X RSubOut4, Pt0, PT, RSubFirst4, ~RSubSecond4, RZ, Pt0, !PT
    [B------:R-:W-:-:S02]    IMAD.WIDE.U32.X Ro6, RFirst0, RSecond6, RZ, Pt4
    [B------:R-:W-:-:S02]    IADD3.X RSubOut5, Pt0, PT, RSubFirst5, ~RSubSecond5, RZ, Pt0, !PT
    [B------:R-:W-:-:S02]    IMAD.WIDE.U32 Ro4, Pt4, RFirst2, RSecond2, Ro4
    [B------:R-:W-:-:S02]    IADD3.X RSubOut6, Pt0, PT, RSubFirst6, ~RSubSecond6, RZ, Pt0, !PT
    [B------:R-:W-:-:S02]    IMAD.WIDE.U32.X Ro6, Pt4, RFirst1.reuse, RSecond5, Ro6, Pt4
    [B------:R-:W-:-:S02]    IADD3.X RSubOut7, Pt0, PT, RSubFirst7, ~RSubSecond7, RZ, Pt0, !PT
    [B------:R-:W-:-:S02]    IMAD.WIDE.U32.X Rt12, RFirst1, RSecond7, RZ, Pt4
    [B------:R-:W-:Y:S04]    IMAD.WIDE.U32 Ro4, Pt4, RFirst3, RSecond1, Ro4
    [B------:R-:W-:-:S02]    IMAD.WIDE.U32.X Ro6, Pt4, RFirst2.reuse, RSecond4, Ro6, Pt4
    [B------:R-:W-:-:S02] @!Pt0 IADD3.X RSubOut0, Pt1, PT, RSubOut0, 0xFFFFFC2F, RZ, !PT, !PT
    [B------:R-:W-:-:S02]    IMAD.WIDE.U32.X Rt12, Pt4, RFirst2, RSecond6, Rt12, Pt4
    [B------:R-:W-:-:S02] @!Pt0 IADD3.X RSubOut1, Pt1, PT, RSubOut1, 0xFFFFFFFE, RZ, Pt1, !PT
    [B------:R-:W-:-:S02]    IMAD.WIDE.U32.X Rt14, PT, RFirst3.reuse, RSecond7, RZ, Pt4
    [B------:R-:W-:-:S02] @!Pt0 IADD3.X RSubOut2, Pt1, PT, RSubOut2, 0xFFFFFFFF, RZ, Pt1, !PT
    [B------:R-:W-:-:S02]    IMAD.WIDE.U32 Ro4, Pt4, RFirst4, RSecond0, Ro4
    [B------:R-:W-:-:S02] @!Pt0 IADD3.X RSubOut3, Pt1, PT, RSubOut3, 0xFFFFFFFF, RZ, Pt1, !PT
    [B------:R-:W-:-:S02]    IMAD.WIDE.U32.X Ro6, Pt4, RFirst3.reuse, RSecond3, Ro6, Pt4
    [B------:R-:W-:-:S02] @!Pt0 IADD3.X RSubOut4, Pt1, PT, RSubOut4, 0xFFFFFFFF, RZ, Pt1, !PT
    [B------:R-:W-:-:S02]    IMAD.WIDE.U32.X Rt12, Pt4, RFirst3, RSecond5, Rt12, Pt4
    [B------:R-:W-:-:S02] @!Pt0 IADD3.X RSubOut5, Pt1, PT, RSubOut5, 0xFFFFFFFF, RZ, Pt1, !PT
    [B------:R-:W-:-:S02]    IMAD.WIDE.U32.X Rt14, Pt4, RFirst4.reuse, RSecond6, Rt14, Pt4
    [B------:R-:W-:-:S02] @!Pt0 IADD3.X RSubOut6, Pt1, PT, RSubOut6, 0xFFFFFFFF, RZ, Pt1, !PT
    [B------:R-:W-:-:S02]    IMAD.WIDE.U32.X Rt16, RFirst5, RSecond7, RZ, Pt4
    [B------:R-:W-:-:S02] @!Pt0 IADD3.X RSubOut7, PT, PT, RSubOut7, 0xFFFFFFFF, RZ, Pt1, !PT
    [B------:R-:W-:Y:S04]    IMAD.WIDE.U32 Ro6, Pt4, RFirst4.reuse, RSecond2, Ro6
    [B------:R-:W-:Y:S04]    IMAD.WIDE.U32.X Rt12, Pt4, RFirst4, RSecond4, Rt12, Pt4
    [B------:R-:W-:Y:S04]    IMAD.WIDE.U32.X Rt14, Pt4, RFirst5, RSecond5, Rt14, Pt4
    [B------:R-:W-:Y:S04]    IMAD.WIDE.U32.X Rt16, Pt4, RFirst6, RSecond6, Rt16, Pt4
    [B------:R-:W-:-:S01]    IMAD.WIDE.U32.X Rt18, RFirst7, RSecond7, RZ, Pt4
//6: 5B 5D 6E 7F, get C3
    [B------:R-:W-:Y:S04]    IMAD.WIDE.U32 Ro6, Pt4, RFirst5.reuse, RSecond1, Ro6
    [B------:R-:W-:Y:S04]    IMAD.WIDE.U32.X Rt12, Pt4, RFirst5, RSecond3, Rt12, Pt4
    [B------:R-:W-:Y:S04]    IMAD.WIDE.U32.X Rt14, Pt4, RFirst6.reuse, RSecond4, Rt14, Pt4
    [B------:R-:W-:-:S01]    IMAD.WIDE.U32.X Rt16, Pt3, RFirst7, RSecond5, Rt16, Pt4
//6: 6A 6C 7D, get C2
    [B------:R-:W-:Y:S04]    IMAD.WIDE.U32 Ro6, Pt4, RFirst6.reuse, RSecond0, Ro6
    [B------:R-:W-:Y:S04]    IMAD.WIDE.U32.X Rt12, Pt4, RFirst6, RSecond2, Rt12, Pt4
    [B------:R-:W-:-:S01]    IMAD.WIDE.U32.X Rt14, Pt2, RFirst7.reuse, RSecond3, Rt14, Pt4
//8: 7B, get C1
    [B------:R-:W-:-:S01]    IMAD.WIDE.U32 Rt12, Pt1, RFirst7, RSecond1, Rt12
///////////////////////////////
//0: 0B
    [B------:R-:W-:-:S01]    IMAD.WIDE.U32 Rt0, RFirst0, RSecond1, RZ
    [B------:R-:W-:-:S01]    IADD3.X Rt10, RZ, RZ, RZ, Pt1, !PT //C1 to Rt10
//0: 1A 0D
    [B------:R-:W-:Y:S04]    IMAD.WIDE.U32 Rt0, Pt0, RFirst1, RSecond0, Rt0
    [B------:R-:W-:-:S01]    IMAD.WIDE.U32.X Rt2, RFirst0, RSecond3, RZ, Pt0
    [B------:R-:W-:-:S01]    IADD3.X Rt11, RZ, RZ, RZ, Pt2, !PT //C2 to Rt11
//2: 1C 0F
    [B------:R-:W-:-:S03]    IMAD.WIDE.U32 Rt2, Pt0, RFirst1, RSecond2, Rt2
    [B------:R-:W-:-:S01]    IADD3.X Rt18, Pt1, PT, Rt18, RZ, RZ, Pt3, !PT //now use C3 and C7 and forget
    [B------:R-:W-:-:S01]    IMAD.WIDE.U32.X Rt4, RFirst0, RSecond5, RZ, Pt0
    [B------:R-:W-:-:S01]    IADD3 Ro1, Pt3, Ro1, Rt0, RZ //build res
//2: 2B 1E 0H
    [B------:R-:W-:-:S03]    IMAD.WIDE.U32 Rt2, Pt0, RFirst2, RSecond1, Rt2
    [B------:R-:W-:-:S01]    IADD3.X Ro2, Pt3, Ro2, Rt1, RZ, Pt3, !PT //build res
    [B------:R-:W-:-:S04]    IMAD.WIDE.U32.X Rt4, Pt0, RFirst1, RSecond4, Rt4, Pt0
    [B------:R-:W-:-:S01]    IMAD.WIDE.U32.X Rt0, RFirst0, RSecond7, RZ, Pt0
//2: 3A 2D 1G 2H
    [B------:R-:W-:Y:S04]    IMAD.WIDE.U32 Rt2, Pt0, RFirst3, RSecond0, Rt2
    [B------:R-:W-:-:S02]    IMAD.WIDE.U32.X Rt4, Pt0, RFirst2, RSecond3, Rt4, Pt0
    [B------:R-:W-:-:S01]    IADD3.X Ro3, Pt3, Ro3, Rt2, RZ, Pt3, !PT //build res
    [B------:R-:W-:-:S03]    IMAD.WIDE.U32.X Rt0, Pt0, RFirst1, RSecond6, Rt0, Pt0
    [B------:R-:W-:-:S01]    IADD3.X Ro4, Pt3, Ro4, Rt3, RZ, Pt3, !PT //build res
    [B------:R-:W-:-:S01]    IMAD.WIDE.U32.X Rt2, RFirst2, RSecond7, RZ, Pt0
//4: 3C 2F 3G 4H
    [B------:R-:W-:Y:S04]    IMAD.WIDE.U32 Rt4, Pt0, RFirst3, RSecond2, Rt4
    [B------:R-:W-:Y:S04]    IMAD.WIDE.U32.X Rt0, Pt0, RFirst2, RSecond5, Rt0, Pt0
    [B------:R-:W-:Y:S04]    IMAD.WIDE.U32.X Rt2, Pt0, RFirst3, RSecond6, Rt2, Pt0
    [B------:R-:W-:-:S01]    IMAD.WIDE.U32.X Rt6, RFirst4, RSecond7, RZ, Pt0
//4: 4B 3E 4F 5G 6H
    [B------:R-:W-:Y:S04]    IMAD.WIDE.U32 Rt4, Pt0, RFirst4, RSecond1, Rt4
    [B------:R-:W-:Y:S04]    IMAD.WIDE.U32.X Rt0, Pt0, RFirst3, RSecond4, Rt0, Pt0
    [B------:R-:W-:Y:S04]    IMAD.WIDE.U32.X Rt2, Pt0, RFirst4, RSecond5, Rt2, Pt0
    [B------:R-:W-:Y:S04]    IMAD.WIDE.U32.X Rt6, Pt0, RFirst5, RSecond6, Rt6, Pt0
    [B------:R-:W-:-:S01]    IMAD.WIDE.U32.X Rt8, RFirst6, RSecond7, RZ, Pt0
//4: 5A 4D 5E 6F 7G, get C7(store in Pt4)
    [B------:R-:W-:Y:S04]    IMAD.WIDE.U32 Rt4, Pt0, RFirst5, RSecond0, Rt4
    [B------:R-:W-:-:S02]    IMAD.WIDE.U32.X Rt0, Pt0, RFirst4, RSecond3, Rt0, Pt0
    [B------:R-:W-:-:S01]    IADD3.X Ro5, Pt3, Ro5, Rt4, RZ, Pt3, !PT //build res
    [B------:R-:W-:-:S02]    IMAD.WIDE.U32.X Rt2, Pt0, RFirst5, RSecond4, Rt2, Pt0
    [B------:R-:W-:-:S01]    IADD3.X Ro6, Pt3, Ro6, Rt5, RZ, Pt3, !PT //build res
    [B------:R-:W-:Y:S04]    IMAD.WIDE.U32.X Rt6, Pt0, RFirst6, RSecond5, Rt6, Pt0
    [B------:R-:W-:-:S01]    IMAD.WIDE.U32.X Rt8, Pt4, RFirst7, RSecond6, Rt8, Pt0
//6: 5C 6D 7E, get C6
    [B------:R-:W-:-:S03]    IMAD.WIDE.U32 Rt0, Pt0, RFirst5, RSecond2, Rt0
    [B------:R-:W-:-:S01]    IADD3.X Rt19, PT, PT, Rt19, RZ, RZ, Pt1, Pt4 //now use C3 and C7 and forget
    [B------:R-:W-:Y:S04]    IMAD.WIDE.U32.X Rt2, Pt0, RFirst6, RSecond3, Rt2, Pt0
    [B------:R-:W-:-:S01]    IMAD.WIDE.U32.X Rt6, Pt1, RFirst7, RSecond4, Rt6, Pt0 //C6 to Pt1
//6: 6B 7C, get C5
    [B------:R-:W-:Y:S04]    IMAD.WIDE.U32 Rt0, Pt0, RFirst6, RSecond1, Rt0

    [B------:R-:W-:-:S01]    IMAD.WIDE.U32.X Rt2, Pt2, RFirst7, RSecond2, Rt2, Pt0
//6: 7A, get C4
    [B------:R-:W-:-:S01]    IMAD.WIDE.U32 Rt0, Pt4, RFirst7, RSecond0, Rt0
    [B------:R-:W-:-:S01]    IADD3.X Rt4, RZ, RZ, RZ, Pt2, !PT //C5 to Rt4
    [B------:R-:W-:Y:S04]    IADD3.X Ro7, Pt3, Ro7, Rt0, RZ, Pt3, !PT //build res
    [B------:R-:W-:Y:S04]    IADD3.X Rt12, Pt3, Rt12, Rt1, RZ, Pt3, !PT //build res
    [B------:R-:W-:-:S01]    IADD3.X Rt13, Pt3, Pt4, Rt13, Rt2, RZ, Pt3, Pt4 //build res
    [B------:R-:W-:-:S01]    IMAD.WIDE.U32 Rt0, Rt12, 0x3D1, RZ
    [B------:R-:W-:-:S01]    IADD3.X Rt14, Pt3, Pt4, Rt14, Rt3, Rt10, Pt3, Pt4 //build res
    [B------:R-:W-:-:S01]    IADD3.X Rt10, RZ, RZ, RZ, Pt1, !PT //C6 to Rt10
    [B------:R-:W-:-:S01]    IMAD.WIDE.U32.X Rt2, Pt2, Rt13, 0x3D1, Rt12, !PT
    [B------:R-:W-:-:S01]    IADD3.X Rt15, Pt3, Pt4, Rt15, Rt6, Rt4, Pt3, Pt4 //build res
    [B------:R-:W-:-:S01]    IADD3.X Ro0, Pt0, Pt1, Ro0, Rt0, RZ, !PT, !PT
    [B------:R-:W-:-:S01]    IADD3.X Rt16, Pt3, Pt4, Rt16, Rt7, Rt11, Pt3, Pt4 //build res
    [B------:R-:W-:-:S01]    IMAD.WIDE.U32 Rt4, Rt14, 0x3D1, RZ
    [B------:R-:W-:-:S01]    IADD3.X Rt17, Pt3, Pt4, Rt17, Rt8, Rt10, Pt3, Pt4 //build res
    [B------:R-:W-:-:S01]    IADD3.X Ro1, Pt0, Pt1, Rt1, Rt2, Ro1, Pt0, Pt1
    [B------:R-:W-:-:S01]    IADD3.X Rt18, Pt3, Pt4, Rt18, Rt9, RZ, Pt3, Pt4 //build res
    [B------:R-:W-:-:S01]    IMAD.WIDE.U32.X Rt6, Pt2, Rt15, 0x3D1, Rt14, Pt2
    [B------:R-:W-:-:S01]    IADD3.X Rt19, Rt19, RZ, RZ, Pt3, Pt4 //build res
// reduction
    [B------:R-:W-:-:S01]    IADD3.X Ro2, Pt0, Pt1, Rt3, Rt4, Ro2, Pt0, Pt1
    [B------:R-:W-:-:S01]    IMAD.WIDE.U32 Rt8, Rt16, 0x3D1, RZ
    [B------:R-:W-:-:S01]    IADD3.X Ro3, Pt0, Pt1, Rt5, Rt6, Ro3, Pt0, Pt1
    [B------:R-:W-:-:S01]    IMAD.WIDE.U32.X Rt4, Pt2, Rt17, 0x3D1, Rt16, Pt2
    [B------:R-:W-:-:S01]    IMAD.WIDE.U32 Rt0, Rt18, 0x3D1, RZ
    [B------:R-:W-:-:S01]    IADD3.X Ro4, Pt0, Pt1, Rt7, Rt8, Ro4, Pt0, Pt1
    [B------:R-:W-:-:S02]    IMAD.WIDE.U32.X Rt2, Pt2, Rt19, 0x3D1, Rt18, Pt2
    [B------:R-:W-:-:S02]    IADD3.X Ro5, Pt0, Pt1, Rt9, Rt4, Ro5, Pt0, Pt1
    [B------:R-:W-:-:S02]    IMAD.WIDE.U32 Rpreout0, RSubOut0.reuse, RSecond0, RZ
    [B------:R-:W-:-:S02]    IADD3.X Ro6, Pt0, Pt1, Rt5, Rt0, Ro6, Pt0, Pt1
    [B------:R-:W-:-:S02]    IMAD.WIDE.U32 Rpreout2, RSubOut0, RSecond2, RZ
    [B------:R-:W-:-:S02]    IADD3.X Ro7, Pt0, Pt1, Rt1, Rt2, Ro7, Pt0, Pt1
    [B------:R-:W-:-:S02]    IMAD.WIDE.U32 Rpreout2, Pt4, RSubOut1.reuse, RSecond1, Rpreout2
    [B------:R-:W-:-:S02]    IADD3.X Rt8, Pt0, PT, Rt3, RZ, RZ, Pt0, Pt1
    [B------:R-:W-:-:S02]    IMAD.WIDE.U32.X Rpreout4, RSubOut0, RSecond4, RZ, Pt4
    [B------:R-:W-:-:S02]    IADD3.X Rt9, Pt0, Pt1, RZ, RZ, RZ, Pt0, Pt2
    [B------:R-:W-:-:S02]    IMAD.WIDE.U32 Rpreout2, Pt4, RSubOut2, RSecond0, Rpreout2
    [B------:R-:W-:-:S02]    IMAD.WIDE.U32.X Ro0, Pt3, Rt8, 0x3D1, Ro0, !PT
    [B------:R-:W-:-:S02]    IMAD.WIDE.U32 Rt2, Rt9, 0x3D1, Rt8 //no carry occurs!
    [B------:R-:W-:-:S02]    IMAD.WIDE.U32.X Rpreout4, Pt4, RSubOut1.reuse, RSecond3, Rpreout4, Pt4
    [B------:R-:W-:-:S02]    IADD3.X Ro1, Pt0, PT, Ro1, Rt2, RZ, !PT, !PT
    [B------:R-:W-:-:S02]    IMAD.WIDE.U32.X Rpreout6, RSubOut0, RSecond6, RZ, Pt4
    [B------:R-:W-:-:S02]    IADD3.X Ro2, Pt0, Pt1, Rt3, Ro2, RZ, Pt0, Pt3
    [B------:R-:W-:-:S02]    IMAD.WIDE.U32 Rpreout4, Pt4, RSubOut2, RSecond2, Rpreout4
    [B------:R-:W-:-:S02]    IADD3.X Ro3, Pt0, PT, Ro3, RZ, RZ, Pt0, Pt1
    [B------:R-:W-:-:S02]    IMAD.WIDE.U32.X Rpreout6, Pt4, RSubOut1.reuse, RSecond5, Rpreout6, Pt4
    [B------:R-:W-:-:S02]    IADD3.X Ro4, Pt0, PT, Ro4, RZ, RZ, Pt0, !PT
    [B------:R-:W-:-:S02]    IMAD.WIDE.U32.X Rt12, RSubOut1, RSecond7, RZ, Pt4
    [B------:R-:W-:-:S02]    IADD3.X Ro5, Pt0, PT, Ro5, RZ, RZ, Pt0, !PT
    [B------:R-:W-:-:S02]    IMAD.WIDE.U32 Rpreout4, Pt4, RSubOut3, RSecond1, Rpreout4
    [B------:R-:W-:-:S02]    IADD3.X Ro6, Pt0, PT, Ro6, RZ, RZ, Pt0, !PT
    [B------:R-:W-:-:S02]    IMAD.WIDE.U32.X Rpreout6, Pt4, RSubOut2.reuse, RSecond4, Rpreout6, Pt4
    [B------:R-:W-:-:S02]    IADD3.X Ro7, PT, PT, Ro7, RZ, RZ, Pt0, !PT
// second mulmod /////////

    [B------:R-:W-:Y:S04]    IMAD.WIDE.U32.X Rt12, Pt4, RSubOut2, RSecond6, Rt12, Pt4
    [B------:R-:W-:-:S01]    IMAD.WIDE.U32.X Rt14, PT, RSubOut3.reuse, RSecond7, RZ, Pt4
//4: 4A 3D 3F 4G 5H
    [B------:R-:W-:Y:S04]    IMAD.WIDE.U32 Rpreout4, Pt4, RSubOut4, RSecond0, Rpreout4
    [B------:R-:W-:Y:S04]    IMAD.WIDE.U32.X Rpreout6, Pt4, RSubOut3.reuse, RSecond3, Rpreout6, Pt4
    [B------:R-:W-:Y:S04]    IMAD.WIDE.U32.X Rt12, Pt4, RSubOut3, RSecond5, Rt12, Pt4
    [B------:R-:W-:Y:S04]    IMAD.WIDE.U32.X Rt14, Pt4, RSubOut4.reuse, RSecond6, Rt14, Pt4
    [B------:R-:W-:-:S01]    IMAD.WIDE.U32.X Rt16, RSubOut5, RSecond7, RZ, Pt4
//6: 4C 4E 5F 6G 7H
    [B------:R-:W-:Y:S04]    IMAD.WIDE.U32 Rpreout6, Pt4, RSubOut4.reuse, RSecond2, Rpreout6
    [B------:R-:W-:Y:S04]    IMAD.WIDE.U32.X Rt12, Pt4, RSubOut4, RSecond4, Rt12, Pt4
    [B------:R-:W-:Y:S04]    IMAD.WIDE.U32.X Rt14, Pt4, RSubOut5, RSecond5, Rt14, Pt4
    [B------:R-:W-:Y:S04]    IMAD.WIDE.U32.X Rt16, Pt4, RSubOut6, RSecond6, Rt16, Pt4
    [B------:R-:W-:-:S01]    IMAD.WIDE.U32.X Rt18, RSubOut7, RSecond7, RZ, Pt4
//6: 5B 5D 6E 7F, get C3
    [B------:R-:W-:Y:S04]    IMAD.WIDE.U32 Rpreout6, Pt4, RSubOut5.reuse, RSecond1, Rpreout6
    [B------:R-:W-:Y:S04]    IMAD.WIDE.U32.X Rt12, Pt4, RSubOut5, RSecond3, Rt12, Pt4
    [B------:R-:W-:Y:S04]    IMAD.WIDE.U32.X Rt14, Pt4, RSubOut6.reuse, RSecond4, Rt14, Pt4
    [B------:R-:W-:-:S01]    IMAD.WIDE.U32.X Rt16, Pt3, RSubOut7, RSecond5, Rt16, Pt4
//6: 6A 6C 7D, get C2
    [B------:R-:W-:Y:S04]    IMAD.WIDE.U32 Rpreout6, Pt4, RSubOut6.reuse, RSecond0, Rpreout6
    [B------:R-:W-:Y:S04]    IMAD.WIDE.U32.X Rt12, Pt4, RSubOut6, RSecond2, Rt12, Pt4
    [B------:R-:W-:-:S01]    IMAD.WIDE.U32.X Rt14, Pt2, RSubOut7.reuse, RSecond3, Rt14, Pt4
//8: 7B, get C1
    [B------:R-:W-:-:S01]    IMAD.WIDE.U32 Rt12, Pt1, RSubOut7, RSecond1, Rt12
///////////////////////////////
//0: 0B
    [B------:R-:W-:-:S01]    IMAD.WIDE.U32 Rt0, RSubOut0, RSecond1, RZ
    [B------:R-:W-:-:S01]    IADD3.X Rt10, RZ, RZ, RZ, Pt1, !PT //C1 to Rt10
//0: 1A 0D
    [B------:R-:W-:Y:S04]    IMAD.WIDE.U32 Rt0, Pt0, RSubOut1, RSecond0, Rt0
    [B------:R-:W-:-:S01]    IMAD.WIDE.U32.X Rt2, RSubOut0, RSecond3, RZ, Pt0
    [B------:R-:W-:-:S01]    IADD3.X Rt11, RZ, RZ, RZ, Pt2, !PT //C2 to Rt11
//2: 1C 0F
    [B------:R-:W-:-:S03]    IMAD.WIDE.U32 Rt2, Pt0, RSubOut1, RSecond2, Rt2
    [B------:R-:W-:-:S01]    IADD3.X Rt18, Pt1, PT, Rt18, RZ, RZ, Pt3, !PT //now use C3 and C7 and forget
    [B------:R-:W-:-:S01]    IMAD.WIDE.U32.X Rt4, RSubOut0, RSecond5, RZ, Pt0
    [B------:R-:W-:-:S01]    IADD3 Rpreout1, Pt3, Rpreout1, Rt0, RZ //build res
//2: 2B 1E 0H
    [B------:R-:W-:-:S03]    IMAD.WIDE.U32 Rt2, Pt0, RSubOut2, RSecond1, Rt2
    [B------:R-:W-:-:S01]    IADD3.X Rpreout2, Pt3, Rpreout2, Rt1, RZ, Pt3, !PT //build res
    [B------:R-:W-:-:S04]    IMAD.WIDE.U32.X Rt4, Pt0, RSubOut1, RSecond4, Rt4, Pt0
    [B------:R-:W-:-:S01]    IMAD.WIDE.U32.X Rt0, RSubOut0, RSecond7, RZ, Pt0
//2: 3A 2D 1G 2H
    [B------:R-:W-:Y:S04]    IMAD.WIDE.U32 Rt2, Pt0, RSubOut3, RSecond0, Rt2
    [B------:R-:W-:-:S02]    IMAD.WIDE.U32.X Rt4, Pt0, RSubOut2, RSecond3, Rt4, Pt0
    [B------:R-:W-:-:S01]    IADD3.X Rpreout3, Pt3, Rpreout3, Rt2, RZ, Pt3, !PT //build res
    [B------:R-:W-:-:S03]    IMAD.WIDE.U32.X Rt0, Pt0, RSubOut1, RSecond6, Rt0, Pt0
    [B------:R-:W-:-:S01]    IADD3.X Rpreout4, Pt3, Rpreout4, Rt3, RZ, Pt3, !PT //build res
    [B------:R-:W-:-:S01]    IMAD.WIDE.U32.X Rt2, RSubOut2, RSecond7, RZ, Pt0
//4: 3C 2F 3G 4H
    [B------:R-:W-:Y:S04]    IMAD.WIDE.U32 Rt4, Pt0, RSubOut3, RSecond2, Rt4
    [B------:R-:W-:Y:S04]    IMAD.WIDE.U32.X Rt0, Pt0, RSubOut2, RSecond5, Rt0, Pt0
    [B------:R-:W-:Y:S04]    IMAD.WIDE.U32.X Rt2, Pt0, RSubOut3, RSecond6, Rt2, Pt0
    [B------:R-:W-:-:S01]    IMAD.WIDE.U32.X Rt6, RSubOut4, RSecond7, RZ, Pt0
//4: 4B 3E 4F 5G 6H
    [B------:R-:W-:Y:S04]    IMAD.WIDE.U32 Rt4, Pt0, RSubOut4, RSecond1, Rt4
    [B------:R-:W-:Y:S04]    IMAD.WIDE.U32.X Rt0, Pt0, RSubOut3, RSecond4, Rt0, Pt0
    [B------:R-:W-:Y:S04]    IMAD.WIDE.U32.X Rt2, Pt0, RSubOut4, RSecond5, Rt2, Pt0
    [B------:R-:W-:Y:S04]    IMAD.WIDE.U32.X Rt6, Pt0, RSubOut5, RSecond6, Rt6, Pt0
    [B------:R-:W-:-:S01]    IMAD.WIDE.U32.X Rt8, RSubOut6, RSecond7, RZ, Pt0
//4: 5A 4D 5E 6F 7G, get C7(store in Pt4)
    [B------:R-:W-:Y:S04]    IMAD.WIDE.U32 Rt4, Pt0, RSubOut5, RSecond0, Rt4
    [B------:R-:W-:-:S02]    IMAD.WIDE.U32.X Rt0, Pt0, RSubOut4, RSecond3, Rt0, Pt0
    [B------:R-:W-:-:S01]    IADD3.X Rpreout5, Pt3, Rpreout5, Rt4, RZ, Pt3, !PT //build res
    [B------:R-:W-:-:S02]    IMAD.WIDE.U32.X Rt2, Pt0, RSubOut5, RSecond4, Rt2, Pt0
    [B------:R-:W-:-:S01]    IADD3.X Rpreout6, Pt3, Rpreout6, Rt5, RZ, Pt3, !PT //build res
    [B------:R-:W-:Y:S04]    IMAD.WIDE.U32.X Rt6, Pt0, RSubOut6, RSecond5, Rt6, Pt0
    [B------:R-:W-:-:S01]    IMAD.WIDE.U32.X Rt8, Pt4, RSubOut7, RSecond6, Rt8, Pt0
//6: 5C 6D 7E, get C6
    [B------:R-:W-:-:S03]    IMAD.WIDE.U32 Rt0, Pt0, RSubOut5, RSecond2, Rt0
    [B------:R-:W-:-:S01]    IADD3.X Rt19, PT, PT, Rt19, RZ, RZ, Pt1, Pt4 //now use C3 and C7 and forget
    [B------:R-:W-:Y:S04]    IMAD.WIDE.U32.X Rt2, Pt0, RSubOut6, RSecond3, Rt2, Pt0
    [B------:R-:W-:-:S01]    IMAD.WIDE.U32.X Rt6, Pt1, RSubOut7, RSecond4, Rt6, Pt0 //C6 to Pt1
//6: 6B 7C, get C5
    [B------:R-:W-:Y:S04]    IMAD.WIDE.U32 Rt0, Pt0, RSubOut6, RSecond1, Rt0

    [B------:R-:W-:-:S01]    IMAD.WIDE.U32.X Rt2, Pt2, RSubOut7, RSecond2, Rt2, Pt0
//6: 7A, get C4
    [B------:R-:W-:-:S02]    IMAD.WIDE.U32 Rt0, Pt4, RSubOut7, RSecond0, Rt0
    [B------:R-:W-:-:S01]    IADD3.X Rt4, RZ, RZ, RZ, Pt2, !PT //C5 to Rt4
    [B------:R-:W-:Y:S04]    IADD3.X Rpreout7, Pt3, Rpreout7, Rt0, RZ, Pt3, !PT //build res
    [B------:R-:W-:Y:S04]    IADD3.X Rt12, Pt3, Rt12, Rt1, RZ, Pt3, !PT //build res
    [B------:R-:W-:-:S01]    IADD3.X Rt13, Pt3, Pt4, Rt13, Rt2, RZ, Pt3, Pt4 //build res
    [B------:R-:W-:-:S02]    IMAD.WIDE.U32 Rt0, Rt12, 0x3D1, RZ
    [B------:R-:W-:-:S01]    IADD3.X Rt14, Pt3, Pt4, Rt14, Rt3, Rt10, Pt3, Pt4 //build res
    [B------:R-:W-:-:S01]    IADD3.X Rt10, RZ, RZ, RZ, Pt1, !PT //C6 to Rt10
    [B------:R-:W-:-:S02]    IMAD.WIDE.U32.X Rt2, Pt2, Rt13, 0x3D1, Rt12, !PT
    [B------:R-:W-:-:S01]    IADD3.X Rt15, Pt3, Pt4, Rt15, Rt6, Rt4, Pt3, Pt4 //build res
    [B------:R-:W-:-:S01]    IADD3.X Rpreout0, Pt0, Pt1, Rpreout0, Rt0, RZ, !PT, !PT
    [B------:R-:W-:-:S01]    IADD3.X Rt16, Pt3, Pt4, Rt16, Rt7, Rt11, Pt3, Pt4 //build res
    [B------:R-:W-:-:S02]    IMAD.WIDE.U32 Rt4, Rt14, 0x3D1, RZ
    [B------:R-:W-:-:S01]    IADD3.X Rt17, Pt3, Pt4, Rt17, Rt8, Rt10, Pt3, Pt4 //build res
    [B------:R-:W-:-:S01]    IADD3.X Rpreout1, Pt0, Pt1, Rt1, Rt2, Rpreout1, Pt0, Pt1
    [B------:R-:W-:-:S01]    IADD3.X Rt18, Pt3, Pt4, Rt18, Rt9, RZ, Pt3, Pt4 //build res
    [B------:R-:W-:-:S02]    IMAD.WIDE.U32.X Rt6, Pt2, Rt15, 0x3D1, Rt14, Pt2
    [B------:R-:W-:-:S01]    IADD3.X Rt19, Rt19, RZ, RZ, Pt3, Pt4 //build res
// reduction
    [B------:R-:W-:-:S01]    IADD3.X Rpreout2, Pt0, Pt1, Rt3, Rt4, Rpreout2, Pt0, Pt1
    [B------:R-:W-:-:S02]    IMAD.WIDE.U32 Rt8, Rt16, 0x3D1, RZ
    [B------:R-:W-:-:S01]    IADD3.X Rpreout3, Pt0, Pt1, Rt5, Rt6, Rpreout3, Pt0, Pt1
    [B------:R-:W-:-:S02]    IMAD.WIDE.U32.X Rt4, Pt2, Rt17, 0x3D1, Rt16, Pt2
    [B------:R-:W-:-:S02]    IMAD.WIDE.U32 Rt0, Rt18, 0x3D1, RZ
    [B------:R-:W-:-:S01]    IADD3.X Rpreout4, Pt0, Pt1, Rt7, Rt8, Rpreout4, Pt0, Pt1
    [B------:R-:W-:-:S02]    IMAD.WIDE.U32.X Rt2, Pt2, Rt19, 0x3D1, Rt18, Pt2
    [B------:R-:W-:Y:S04]    IADD3.X Rpreout5, Pt0, Pt1, Rt9, Rt4, Rpreout5, Pt0, Pt1
    [B------:R-:W-:Y:S04]    IADD3.X Rpreout6, Pt0, Pt1, Rt5, Rt0, Rpreout6, Pt0, Pt1
    [B------:R-:W-:Y:S04]    IADD3.X Rpreout7, Pt0, Pt1, Rt1, Rt2, Rpreout7, Pt0, Pt1
//Rt8..Rt9 is "tmp[4] in cpu code, Rt8 <= 0x3d1 (checked P-1*P-1), Rt9 <= 1
    [B------:R-:W-:Y:S04]    IADD3.X Rt8, Pt0, PT, Rt3, RZ, RZ, Pt0, Pt1
    [B------:R-:W-:Y:S01]    IADD3.X Rt9, Pt0, Pt1, RZ, RZ, RZ, Pt0, Pt2
////
    [B------:R-:W-:-:S02]    IMAD.WIDE.U32.X Rout0, Pt3, Rt8, 0x3D1, Rpreout0, !PT
    [B------:R-:W-:-:S03]    IMAD.WIDE.U32 Rt2, Rt9, 0x3D1, Rt8 //no carry occurs!
    [B------:R-:W-:Y:S04]    IADD3.X Rout1, Pt0, PT, Rout1, Rt2, RZ, !PT, !PT
    [B------:R-:W-:Y:S04]    IADD3.X Rout2, Pt0, Pt1, Rt3, Rpreout2, RZ, Pt0, Pt3
    [B------:R-:W-:Y:S04]    IADD3.X Rout3, Pt0, PT, Rpreout3, RZ, RZ, Pt0, Pt1
    [B------:R-:W-:Y:S04]    IADD3.X Rout4, Pt0, PT, Rpreout4, RZ, RZ, Pt0, !PT
    [B------:R-:W-:Y:S04]    IADD3.X Rout5, Pt0, PT, Rpreout5, RZ, RZ, Pt0, !PT
    [B------:R-:W-:Y:S04]    IADD3.X Rout6, Pt0, PT, Rpreout6, RZ, RZ, Pt0, !PT
    [B------:R-:W-:Y:S04]    IADD3.X Rout7, PT, PT, Rpreout7, RZ, RZ, Pt0, !PT
}


FUNCTION CalcNewPoints_FusedB()
{
    [B------:R-:W-:-:S04]    IADD3.X RSubOut0, Pt0, PT, RSubFirst0, ~RSubSecond0, RZ, !PT, PT
    [B------:R-:W-:-:S04]    IADD3.X RSubOut1, Pt0, PT, RSubFirst1, ~RSubSecond1, RZ, Pt0, !PT
    [B------:R-:W-:-:S04]    IADD3.X RSubOut2, Pt0, PT, RSubFirst2, ~RSubSecond2, RZ, Pt0, !PT
    [B------:R-:W-:-:S04]    IADD3.X RSubOut3, Pt0, PT, RSubFirst3, ~RSubSecond3, RZ, Pt0, !PT
    [B------:R-:W-:-:S04]    IADD3.X RSubOut4, Pt0, PT, RSubFirst4, ~RSubSecond4, RZ, Pt0, !PT
    [B------:R-:W-:-:S04]    IADD3.X RSubOut5, Pt0, PT, RSubFirst5, ~RSubSecond5, RZ, Pt0, !PT
    [B------:R-:W-:-:S04]    IADD3.X RSubOut6, Pt0, PT, RSubFirst6, ~RSubSecond6, RZ, Pt0, !PT
    [B------:R-:W-:Y:S13]    IADD3.X RSubOut7, Pt0, PT, RSubFirst7, ~RSubSecond7, RZ, Pt0, !PT
//add P if <0
    [B------:R-:W-:-:S04] @!Pt0 IADD3.X RSubOut0, Pt1, PT, RSubOut0, 0xFFFFFC2F, RZ, !PT, !PT
    [B------:R-:W-:-:S02] @!Pt0 IADD3.X RSubOut1, Pt1, PT, RSubOut1, 0xFFFFFFFE, RZ, Pt1, !PT

    [B------:R-:W-:-:S02]    IMAD.WIDE.U32 Ro0, RFirst0, RSecond0, RZ
    [B------:R-:W-:-:S02]    IMAD.WIDE.U32 Ro2, RFirst0, RSecond2, RZ
//2: 1B 0E
    [B------:R-:W-:-:S02]    IMAD.WIDE.U32 Ro2, Pt4, RFirst1, RSecond1, Ro2
    [B------:R-:W-:-:S02] @!Pt0 IADD3.X RSubOut2, Pt1, PT, RSubOut2, 0xFFFFFFFF, RZ, Pt1, !PT

    [B------:R-:W-:-:S02]    IMAD.WIDE.U32.X Ro4, RFirst0, RSecond4, RZ, Pt4
//2: 2A 1D 0G
    [B------:R-:W-:-:S02]    IMAD.WIDE.U32 Ro2, Pt4, RFirst2, RSecond0, Ro2
    [B------:R-:W-:-:S02] @!Pt0 IADD3.X RSubOut3, Pt1, PT, RSubOut3, 0xFFFFFFFF, RZ, Pt1, !PT
    [B------:R-:W-:-:S02]    IMAD.WIDE.U32.X Ro4, Pt4, RFirst1, RSecond3, Ro4, Pt4
    [B------:R-:W-:-:S02] @!Pt0 IADD3.X RSubOut4, Pt1, PT, RSubOut4, 0xFFFFFFFF, RZ, Pt1, !PT
    [B------:R-:W-:-:S02]    IMAD.WIDE.U32.X Ro6, RFirst0, RSecond6, RZ, Pt4
//4: 2C 1F 1H
    [B------:R-:W-:-:S02]    IMAD.WIDE.U32 Ro4, Pt4, RFirst2, RSecond2, Ro4
    [B------:R-:W-:-:S02] @!Pt0 IADD3.X RSubOut5, Pt1, PT, RSubOut5, 0xFFFFFFFF, RZ, Pt1, !PT
    [B------:R-:W-:-:S02]    IMAD.WIDE.U32.X Ro6, Pt4, RFirst1, RSecond5, Ro6, Pt4
    [B------:R-:W-:-:S02] @!Pt0 IADD3.X RSubOut6, Pt1, PT, RSubOut6, 0xFFFFFFFF, RZ, Pt1, !PT
    [B------:R-:W-:-:S02]    IMAD.WIDE.U32.X Rt12, RFirst1, RSecond7, RZ, Pt4
//4: 3B 2E 2G 3H
    [B------:R-:W-:-:S02]    IMAD.WIDE.U32 Ro4, Pt4, RFirst3, RSecond1, Ro4
    [B------:R-:W-:-:S02] @!Pt0 IADD3.X RSubOut7, PT, PT, RSubOut7, 0xFFFFFFFF, RZ, Pt1, !PT
    [B------:R-:W-:Y:S04]    IMAD.WIDE.U32.X Ro6, Pt4, RFirst2, RSecond4, Ro6, Pt4
    [B------:R-:W-:Y:S04]    IMAD.WIDE.U32.X Rt12, Pt4, RFirst2, RSecond6, Rt12, Pt4
    [B------:R-:W-:-:S02]    IMAD.WIDE.U32.X Rt14, PT, RFirst3, RSecond7, RZ, Pt4
//4: 4A 3D 3F 4G 5H
    [B------:R-:W-:Y:S04]    IMAD.WIDE.U32 Ro4, Pt4, RFirst4, RSecond0, Ro4
    [B------:R-:W-:Y:S04]    IMAD.WIDE.U32.X Ro6, Pt4, RFirst3.reuse, RSecond3, Ro6, Pt4
    [B------:R-:W-:Y:S04]    IMAD.WIDE.U32.X Rt12, Pt4, RFirst3, RSecond5, Rt12, Pt4
    [B------:R-:W-:Y:S04]    IMAD.WIDE.U32.X Rt14, Pt4, RFirst4.reuse, RSecond6, Rt14, Pt4
    [B------:R-:W-:-:S02]    IMAD.WIDE.U32.X Rt16, RFirst5, RSecond7, RZ, Pt4
//6: 4C 4E 5F 6G 7H
    [B------:R-:W-:Y:S04]    IMAD.WIDE.U32 Ro6, Pt4, RFirst4.reuse, RSecond2, Ro6
    [B------:R-:W-:Y:S04]    IMAD.WIDE.U32.X Rt12, Pt4, RFirst4, RSecond4, Rt12, Pt4
    [B------:R-:W-:Y:S04]    IMAD.WIDE.U32.X Rt14, Pt4, RFirst5, RSecond5, Rt14, Pt4
    [B------:R-:W-:Y:S04]    IMAD.WIDE.U32.X Rt16, Pt4, RFirst6, RSecond6, Rt16, Pt4
    [B------:R-:W-:-:S02]    IMAD.WIDE.U32.X Rt18, RFirst7, RSecond7, RZ, Pt4
//6: 5B 5D 6E 7F, get C3
    [B------:R-:W-:Y:S04]    IMAD.WIDE.U32 Ro6, Pt4, RFirst5.reuse, RSecond1, Ro6
    [B------:R-:W-:Y:S04]    IMAD.WIDE.U32.X Rt12, Pt4, RFirst5, RSecond3, Rt12, Pt4
    [B------:R-:W-:Y:S04]    IMAD.WIDE.U32.X Rt14, Pt4, RFirst6.reuse, RSecond4, Rt14, Pt4
    [B------:R-:W-:-:S02]    IMAD.WIDE.U32.X Rt16, Pt3, RFirst7, RSecond5, Rt16, Pt4
//6: 6A 6C 7D, get C2
    [B------:R-:W-:Y:S04]    IMAD.WIDE.U32 Ro6, Pt4, RFirst6.reuse, RSecond0, Ro6
    [B------:R-:W-:Y:S04]    IMAD.WIDE.U32.X Rt12, Pt4, RFirst6, RSecond2, Rt12, Pt4
    [B------:R-:W-:-:S02]    IMAD.WIDE.U32.X Rt14, Pt2, RFirst7.reuse, RSecond3, Rt14, Pt4
//8: 7B, get C1
    [B------:R-:W-:-:S02]    IMAD.WIDE.U32 Rt12, Pt1, RFirst7, RSecond1, Rt12
///////////////////////////////
//0: 0B
    [B------:R-:W-:-:S02]    IMAD.WIDE.U32 Rt0, RFirst0, RSecond1, RZ
    [B------:R-:W-:-:S01]    IADD3.X Rt10, RZ, RZ, RZ, Pt1, !PT //C1 to Rt10
//0: 1A 0D
    [B------:R-:W-:Y:S04]    IMAD.WIDE.U32 Rt0, Pt0, RFirst1, RSecond0, Rt0
    [B------:R-:W-:-:S02]    IMAD.WIDE.U32.X Rt2, RFirst0, RSecond3, RZ, Pt0
    [B------:R-:W-:-:S01]    IADD3.X Rt11, RZ, RZ, RZ, Pt2, !PT //C2 to Rt11
//2: 1C 0F
    [B------:R-:W-:-:S03]    IMAD.WIDE.U32 Rt2, Pt0, RFirst1, RSecond2, Rt2
    [B------:R-:W-:-:S01]    IADD3.X Rt18, Pt1, PT, Rt18, RZ, RZ, Pt3, !PT //now use C3 and C7 and forget
    [B------:R-:W-:-:S02]    IMAD.WIDE.U32.X Rt4, RFirst0, RSecond5, RZ, Pt0
    [B------:R-:W-:-:S01]    IADD3 Ro1, Pt3, Ro1, Rt0, RZ //build res
//2: 2B 1E 0H
    [B------:R-:W-:-:S03]    IMAD.WIDE.U32 Rt2, Pt0, RFirst2, RSecond1, Rt2
    [B------:R-:W-:-:S01]    IADD3.X Ro2, Pt3, Ro2, Rt1, RZ, Pt3, !PT //build res
    [B------:R-:W-:-:S04]    IMAD.WIDE.U32.X Rt4, Pt0, RFirst1, RSecond4, Rt4, Pt0
    [B------:R-:W-:-:S02]    IMAD.WIDE.U32.X Rt0, RFirst0, RSecond7, RZ, Pt0
//2: 3A 2D 1G 2H
    [B------:R-:W-:Y:S04]    IMAD.WIDE.U32 Rt2, Pt0, RFirst3, RSecond0, Rt2
    [B------:R-:W-:-:S02]    IMAD.WIDE.U32.X Rt4, Pt0, RFirst2, RSecond3, Rt4, Pt0
    [B------:R-:W-:-:S01]    IADD3.X Ro3, Pt3, Ro3, Rt2, RZ, Pt3, !PT //build res
    [B------:R-:W-:-:S03]    IMAD.WIDE.U32.X Rt0, Pt0, RFirst1, RSecond6, Rt0, Pt0
    [B------:R-:W-:-:S01]    IADD3.X Ro4, Pt3, Ro4, Rt3, RZ, Pt3, !PT //build res
    [B------:R-:W-:-:S02]    IMAD.WIDE.U32.X Rt2, RFirst2, RSecond7, RZ, Pt0
//4: 3C 2F 3G 4H
    [B------:R-:W-:Y:S04]    IMAD.WIDE.U32 Rt4, Pt0, RFirst3, RSecond2, Rt4
    [B------:R-:W-:Y:S04]    IMAD.WIDE.U32.X Rt0, Pt0, RFirst2, RSecond5, Rt0, Pt0
    [B------:R-:W-:Y:S04]    IMAD.WIDE.U32.X Rt2, Pt0, RFirst3, RSecond6, Rt2, Pt0
    [B------:R-:W-:-:S02]    IMAD.WIDE.U32.X Rt6, RFirst4, RSecond7, RZ, Pt0
//4: 4B 3E 4F 5G 6H
    [B------:R-:W-:Y:S04]    IMAD.WIDE.U32 Rt4, Pt0, RFirst4, RSecond1, Rt4
    [B------:R-:W-:Y:S04]    IMAD.WIDE.U32.X Rt0, Pt0, RFirst3, RSecond4, Rt0, Pt0
    [B------:R-:W-:Y:S04]    IMAD.WIDE.U32.X Rt2, Pt0, RFirst4, RSecond5, Rt2, Pt0
    [B------:R-:W-:Y:S04]    IMAD.WIDE.U32.X Rt6, Pt0, RFirst5, RSecond6, Rt6, Pt0
    [B------:R-:W-:-:S02]    IMAD.WIDE.U32.X Rt8, RFirst6, RSecond7, RZ, Pt0
//4: 5A 4D 5E 6F 7G, get C7(store in Pt4)
    [B------:R-:W-:Y:S04]    IMAD.WIDE.U32 Rt4, Pt0, RFirst5, RSecond0, Rt4
    [B------:R-:W-:-:S02]    IMAD.WIDE.U32.X Rt0, Pt0, RFirst4, RSecond3, Rt0, Pt0
    [B------:R-:W-:-:S01]    IADD3.X Ro5, Pt3, Ro5, Rt4, RZ, Pt3, !PT //build res
    [B------:R-:W-:-:S02]    IMAD.WIDE.U32.X Rt2, Pt0, RFirst5, RSecond4, Rt2, Pt0
    [B------:R-:W-:-:S01]    IADD3.X Ro6, Pt3, Ro6, Rt5, RZ, Pt3, !PT //build res
    [B------:R-:W-:Y:S04]    IMAD.WIDE.U32.X Rt6, Pt0, RFirst6, RSecond5, Rt6, Pt0
    [B------:R-:W-:-:S02]    IMAD.WIDE.U32.X Rt8, Pt4, RFirst7, RSecond6, Rt8, Pt0
//6: 5C 6D 7E, get C6
    [B------:R-:W-:-:S03]    IMAD.WIDE.U32 Rt0, Pt0, RFirst5, RSecond2, Rt0
    [B------:R-:W-:-:S01]    IADD3.X Rt19, PT, PT, Rt19, RZ, RZ, Pt1, Pt4 //now use C3 and C7 and forget
    [B------:R-:W-:Y:S04]    IMAD.WIDE.U32.X Rt2, Pt0, RFirst6, RSecond3, Rt2, Pt0
    [B------:R-:W-:-:S02]    IMAD.WIDE.U32.X Rt6, Pt1, RFirst7, RSecond4, Rt6, Pt0 //C6 to Pt1
//6: 6B 7C, get C5
    [B------:R-:W-:Y:S04]    IMAD.WIDE.U32 Rt0, Pt0, RFirst6, RSecond1, Rt0

    [B------:R-:W-:-:S02]    IMAD.WIDE.U32.X Rt2, Pt2, RFirst7, RSecond2, Rt2, Pt0
//6: 7A, get C4
    [B------:R-:W-:-:S02]    IMAD.WIDE.U32 Rt0, Pt4, RFirst7, RSecond0, Rt0
    [B------:R-:W-:-:S01]    IADD3.X Rt4, RZ, RZ, RZ, Pt2, !PT //C5 to Rt4
    [B------:R-:W-:Y:S04]    IADD3.X Ro7, Pt3, Ro7, Rt0, RZ, Pt3, !PT //build res
    [B------:R-:W-:Y:S04]    IADD3.X Rt12, Pt3, Rt12, Rt1, RZ, Pt3, !PT //build res
    [B------:R-:W-:-:S01]    IADD3.X Rt13, Pt3, Pt4, Rt13, Rt2, RZ, Pt3, Pt4 //build res
    [B------:R-:W-:-:S02]    IMAD.WIDE.U32 Rt0, Rt12, 0x3D1, RZ
    [B------:R-:W-:-:S01]    IADD3.X Rt14, Pt3, Pt4, Rt14, Rt3, Rt10, Pt3, Pt4 //build res
    [B------:R-:W-:-:S01]    IADD3.X Rt10, RZ, RZ, RZ, Pt1, !PT //C6 to Rt10
    [B------:R-:W-:-:S02]    IMAD.WIDE.U32.X Rt2, Pt2, Rt13, 0x3D1, Rt12, !PT
    [B------:R-:W-:-:S01]    IADD3.X Rt15, Pt3, Pt4, Rt15, Rt6, Rt4, Pt3, Pt4 //build res
    [B------:R-:W-:-:S01]    IADD3.X Ro0, Pt0, Pt1, Ro0, Rt0, RZ, !PT, !PT
    [B------:R-:W-:-:S01]    IADD3.X Rt16, Pt3, Pt4, Rt16, Rt7, Rt11, Pt3, Pt4 //build res
    [B------:R-:W-:-:S02]    IMAD.WIDE.U32 Rt4, Rt14, 0x3D1, RZ
    [B------:R-:W-:-:S01]    IADD3.X Rt17, Pt3, Pt4, Rt17, Rt8, Rt10, Pt3, Pt4 //build res
    [B------:R-:W-:-:S01]    IADD3.X Ro1, Pt0, Pt1, Rt1, Rt2, Ro1, Pt0, Pt1
    [B------:R-:W-:-:S01]    IADD3.X Rt18, Pt3, Pt4, Rt18, Rt9, RZ, Pt3, Pt4 //build res
    [B------:R-:W-:-:S02]    IMAD.WIDE.U32.X Rt6, Pt2, Rt15, 0x3D1, Rt14, Pt2
    [B------:R-:W-:-:S01]    IADD3.X Rt19, Rt19, RZ, RZ, Pt3, Pt4 //build res
// reduction
    [B------:R-:W-:-:S01]    IADD3.X Ro2, Pt0, Pt1, Rt3, Rt4, Ro2, Pt0, Pt1
    [B------:R-:W-:-:S02]    IMAD.WIDE.U32 Rt8, Rt16, 0x3D1, RZ
    [B------:R-:W-:-:S01]    IADD3.X Ro3, Pt0, Pt1, Rt5, Rt6, Ro3, Pt0, Pt1
    [B------:R-:W-:-:S02]    IMAD.WIDE.U32.X Rt4, Pt2, Rt17, 0x3D1, Rt16, Pt2
    [B------:R-:W-:-:S02]    IMAD.WIDE.U32 Rt0, Rt18, 0x3D1, RZ
    [B------:R-:W-:-:S01]    IADD3.X Ro4, Pt0, Pt1, Rt7, Rt8, Ro4, Pt0, Pt1
    [B------:R-:W-:-:S02]    IMAD.WIDE.U32.X Rt2, Pt2, Rt19, 0x3D1, Rt18, Pt2
    [B------:R-:W-:Y:S04]    IADD3.X Ro5, Pt0, Pt1, Rt9, Rt4, Ro5, Pt0, Pt1
    [B------:R-:W-:Y:S04]    IADD3.X Ro6, Pt0, Pt1, Rt5, Rt0, Ro6, Pt0, Pt1
    [B------:R-:W-:Y:S04]    IADD3.X Ro7, Pt0, Pt1, Rt1, Rt2, Ro7, Pt0, Pt1
//Rt8..Rt9 is "tmp[4] in cpu code, Rt8 <= 0x3d1 (checked P-1*P-1), Rt9 <= 1
    [B------:R-:W-:Y:S04]    IADD3.X Rt8, Pt0, PT, Rt3, RZ, RZ, Pt0, Pt1
    [B------:R-:W-:Y:S01]    IADD3.X Rt9, Pt0, Pt1, RZ, RZ, RZ, Pt0, Pt2
////
    [B------:R-:W-:-:S02]    IMAD.WIDE.U32.X Ro0, Pt3, Rt8, 0x3D1, Ro0, !PT
    [B------:R-:W-:-:S03]    IMAD.WIDE.U32 Rt2, Rt9, 0x3D1, Rt8 //no carry occurs!
    [B------:R-:W-:Y:S04]    IADD3.X Ro1, Pt0, PT, Ro1, Rt2, RZ, !PT, !PT
    [B------:R-:W-:Y:S04]    IADD3.X Ro2, Pt0, Pt1, Rt3, Ro2, RZ, Pt0, Pt3
    [B------:R-:W-:Y:S04]    IADD3.X Ro3, Pt0, PT, Ro3, RZ, RZ, Pt0, Pt1
    [B------:R-:W-:Y:S04]    IADD3.X Ro4, Pt0, PT, Ro4, RZ, RZ, Pt0, !PT
    [B------:R-:W-:Y:S04]    IADD3.X Ro5, Pt0, PT, Ro5, RZ, RZ, Pt0, !PT
    [B------:R-:W-:Y:S04]    IADD3.X Ro6, Pt0, PT, Ro6, RZ, RZ, Pt0, !PT
    [B------:R-:W-:Y:S04]    IADD3.X Ro7, PT, PT, Ro7, RZ, RZ, Pt0, !PT
}


FUNCTION CalcNewPoints_FusedC()
{
    [B------:R-:W-:-:S04]    IADD3.X RFirst0, Pt3, PT, RSubFirst0, ~RSubSecond0, RZ, !PT, PT
    [B------:R-:W-:-:S04]    IADD3.X RFirst1, Pt3, PT, RSubFirst1, ~RSubSecond1, RZ, Pt3, !PT
    [B------:R-:W-:-:S04]    IADD3.X RFirst2, Pt3, PT, RSubFirst2, ~RSubSecond2, RZ, Pt3, !PT
    [B------:R-:W-:-:S04]    IADD3.X RFirst3, Pt3, PT, RSubFirst3, ~RSubSecond3, RZ, Pt3, !PT
    [B------:R-:W-:-:S04]    IADD3.X RFirst4, Pt3, PT, RSubFirst4, ~RSubSecond4, RZ, Pt3, !PT
    [B------:R-:W-:-:S04]    IADD3.X RFirst5, Pt3, PT, RSubFirst5, ~RSubSecond5, RZ, Pt3, !PT
    [B------:R-:W-:-:S04]    IADD3.X RFirst6, Pt3, PT, RSubFirst6, ~RSubSecond6, RZ, Pt3, !PT
    [B------:R-:W-:Y:S13]    IADD3.X RFirst7, Pt3, PT, RSubFirst7, ~RSubSecond7, RZ, Pt3, !PT
//add P if <0
    [B------:R-:W-:-:S04] @!Pt3 IADD3.X RFirst0, Pt5, PT, RFirst0, 0xFFFFFC2F, RZ, !PT, !PT
    [B------:R-:W-:-:S01] @!Pt3 IADD3.X RFirst1, Pt5, PT, RFirst1, 0xFFFFFFFE, RZ, Pt5, !PT
//0: 0A 0C
    [B------:R-:W-:-:S01]    IMAD.WIDE.U32 Ro0, Pt0, RFirst0.reuse, RSecond0, Radd0 //Radd0
    [B------:R-:W-:-:S01] @!Pt3 IADD3.X RFirst2, Pt5, PT, RFirst2, 0xFFFFFFFF, RZ, Pt5, !PT
    [B------:R-:W-:-:S01]    IMAD.WIDE.U32 Ro2, Pt1, RFirst0, RSecond2, Radd2 //Radd2
    [B------:R-:W-:-:S01] @!Pt3 IADD3.X RFirst3, Pt5, PT, RFirst3, 0xFFFFFFFF, RZ, Pt5, !PT
//2: 1B 0E
    [B------:R-:W-:-:S02]    IMAD.WIDE.U32 Ro2, Pt4, RFirst1.reuse, RSecond1, Ro2
    [B------:R-:W-:-:S02] @!Pt3 IADD3.X RFirst4, Pt5, PT, RFirst4, 0xFFFFFFFF, RZ, Pt5, !PT
    [B------:R-:W-:-:S02]    IMAD.WIDE.U32.X Ro4, Pt2, RFirst0, RSecond4, Radd4, Pt4 //Radd4
    [B------:R-:W-:-:S02] @!Pt3 IADD3.X RFirst5, Pt5, PT, RFirst5, 0xFFFFFFFF, RZ, Pt5, !PT
//2: 2A 1D 0G
    [B------:R-:W-:-:S02]    IMAD.WIDE.U32 Ro2, Pt4, RFirst2, RSecond0, Ro2, Pt0 //cr0
    [B------:R-:W-:-:S02] @!Pt3 IADD3.X RFirst6, Pt5, PT, RFirst6, 0xFFFFFFFF, RZ, Pt5, !PT
    [B------:R-:W-:-:S02]    IMAD.WIDE.U32.X Ro4, Pt4, RFirst1.reuse, RSecond3, Ro4, Pt4
    [B------:R-:W-:-:S02] @!Pt3 IADD3.X RFirst7, PT, PT, RFirst7, 0xFFFFFFFF, RZ, Pt5, !PT
    [B------:R-:W-:-:S01]    IMAD.WIDE.U32.X Ro6, Pt0, RFirst0, RSecond6, Radd6, Pt4 //Radd6
//4: 2C 1F 1H
    [B------:R-:W-:-:S04]    IMAD.WIDE.U32 Ro4, Pt4, RFirst2, RSecond2, Ro4, Pt1 //cr2
    [B------:R-:W-:-:S04]    IMAD.WIDE.U32.X Ro6, Pt4, RFirst1.reuse, RSecond5, Ro6, Pt4
    [B------:R-:W-:-:S01]    IMAD.WIDE.U32.X Rt12, RFirst1, RSecond7, RZ, Pt4
//4: 3B 2E 2G 3H
    [B------:R-:W-:-:S04]    IMAD.WIDE.U32 Ro4, Pt4, RFirst3, RSecond1, Ro4
    [B------:R-:W-:-:S04]    IMAD.WIDE.U32.X Ro6, Pt4, RFirst2.reuse, RSecond4, Ro6, Pt4
    [B------:R-:W-:-:S04]    IMAD.WIDE.U32.X Rt12, Pt4, RFirst2, RSecond6, Rt12, Pt4
    [B------:R-:W-:-:S01]    IMAD.WIDE.U32.X Rt14, PT, RFirst3.reuse, RSecond7, RZ, Pt4
//4: 4A 3D 3F 4G 5H
    [B------:R-:W-:-:S04]    IMAD.WIDE.U32 Ro4, Pt4, RFirst4, RSecond0, Ro4
    [B------:R-:W-:-:S04]    IMAD.WIDE.U32.X Ro6, Pt4, RFirst3.reuse, RSecond3, Ro6, Pt4
    [B------:R-:W-:-:S04]    IMAD.WIDE.U32.X Rt12, Pt4, RFirst3, RSecond5, Rt12, Pt4
    [B------:R-:W-:-:S04]    IMAD.WIDE.U32.X Rt14, Pt4, RFirst4.reuse, RSecond6, Rt14, Pt4
    [B------:R-:W-:-:S01]    IMAD.WIDE.U32.X Rt16, RFirst5, RSecond7, RZ, Pt4
//6: 4C 4E 5F 6G 7H
    [B------:R-:W-:-:S04]    IMAD.WIDE.U32 Ro6, Pt4, RFirst4.reuse, RSecond2, Ro6, Pt2 //cr4
    [B------:R-:W-:-:S04]    IMAD.WIDE.U32.X Rt12, Pt4, RFirst4, RSecond4, Rt12, Pt4
    [B------:R-:W-:-:S04]    IMAD.WIDE.U32.X Rt14, Pt4, RFirst5, RSecond5, Rt14, Pt4
    [B------:R-:W-:-:S04]    IMAD.WIDE.U32.X Rt16, Pt4, RFirst6, RSecond6, Rt16, Pt4
    [B------:R-:W-:-:S01]    IMAD.WIDE.U32.X Rt18, RFirst7, RSecond7, RZ, Pt4
//6: 5B 5D 6E 7F, get C3
    [B------:R-:W-:-:S04]    IMAD.WIDE.U32 Ro6, Pt4, RFirst5.reuse, RSecond1, Ro6
    [B------:R-:W-:-:S04]    IMAD.WIDE.U32.X Rt12, Pt4, RFirst5, RSecond3, Rt12, Pt4
    [B------:R-:W-:-:S04]    IMAD.WIDE.U32.X Rt14, Pt4, RFirst6.reuse, RSecond4, Rt14, Pt4
    [B------:R-:W-:-:S02]    IMAD.WIDE.U32.X Rt16, Pt3, RFirst7, RSecond5, Rt16, Pt4
//6: 6A 6C 7D, get C2
    [B------:R-:W-:-:S04]    IMAD.WIDE.U32 Ro6, Pt4, RFirst6.reuse, RSecond0, Ro6
    [B------:R-:W-:-:S04]    IMAD.WIDE.U32.X Rt12, Pt4, RFirst6, RSecond2, Rt12, Pt4
    [B------:R-:W-:-:S02]    IMAD.WIDE.U32.X Rt14, Pt2, RFirst7.reuse, RSecond3, Rt14, Pt4
//8: 7B, get C1
    [B------:R-:W-:-:S02]    IMAD.WIDE.U32 Rt12, Pt1, RFirst7, RSecond1, Rt12, Pt0 //cr6
///////////////////////////////
//0: 0B
    [B------:R-:W-:-:S02]    IMAD.WIDE.U32 Rt0, RFirst0, RSecond1, RZ
    [B------:R-:W-:-:S01]    IADD3.X Rt10, RZ, RZ, RZ, Pt1, !PT //C1 to Rt10
//0: 1A 0D
    [B------:R-:W-:-:S04]    IMAD.WIDE.U32 Rt0, Pt0, RFirst1, RSecond0, Rt0
    [B------:R-:W-:-:S02]    IMAD.WIDE.U32.X Rt2, RFirst0, RSecond3, RZ, Pt0
    [B------:R-:W-:-:S01]    IADD3.X Rt11, RZ, RZ, RZ, Pt2, !PT //C2 to Rt11
//2: 1C 0F
    [B------:R-:W-:-:S03]    IMAD.WIDE.U32 Rt2, Pt0, RFirst1, RSecond2, Rt2
    [B------:R-:W-:-:S01]    IADD3.X Rt18, Pt1, PT, Rt18, RZ, RZ, Pt3, !PT //now use C3 and C7 and forget
    [B------:R-:W-:-:S02]    IMAD.WIDE.U32.X Rt4, RFirst0, RSecond5, RZ, Pt0
    [B------:R-:W-:-:S01]    IADD3 Ro1, Pt3, Ro1, Rt0, RZ //build res
//2: 2B 1E 0H
    [B------:R-:W-:-:S03]    IMAD.WIDE.U32 Rt2, Pt0, RFirst2, RSecond1, Rt2
    [B------:R-:W-:-:S01]    IADD3.X Ro2, Pt3, Ro2, Rt1, RZ, Pt3, !PT //build res
    [B------:R-:W-:-:S04]    IMAD.WIDE.U32.X Rt4, Pt0, RFirst1, RSecond4, Rt4, Pt0
    [B------:R-:W-:-:S02]    IMAD.WIDE.U32.X Rt0, RFirst0, RSecond7, RZ, Pt0
//2: 3A 2D 1G 2H
    [B------:R-:W-:-:S04]    IMAD.WIDE.U32 Rt2, Pt0, RFirst3, RSecond0, Rt2
    [B------:R-:W-:-:S02]    IMAD.WIDE.U32.X Rt4, Pt0, RFirst2, RSecond3, Rt4, Pt0
    [B------:R-:W-:-:S01]    IADD3.X Ro3, Pt3, Ro3, Rt2, RZ, Pt3, !PT //build res
    [B------:R-:W-:-:S03]    IMAD.WIDE.U32.X Rt0, Pt0, RFirst1, RSecond6, Rt0, Pt0
    [B------:R-:W-:-:S01]    IADD3.X Ro4, Pt3, Ro4, Rt3, RZ, Pt3, !PT //build res
    [B------:R-:W-:-:S02]    IMAD.WIDE.U32.X Rt2, RFirst2, RSecond7, RZ, Pt0
//4: 3C 2F 3G 4H
    [B------:R-:W-:-:S04]    IMAD.WIDE.U32 Rt4, Pt0, RFirst3, RSecond2, Rt4
    [B------:R-:W-:-:S04]    IMAD.WIDE.U32.X Rt0, Pt0, RFirst2, RSecond5, Rt0, Pt0
    [B------:R-:W-:-:S04]    IMAD.WIDE.U32.X Rt2, Pt0, RFirst3, RSecond6, Rt2, Pt0
    [B------:R-:W-:-:S02]    IMAD.WIDE.U32.X Rt6, RFirst4, RSecond7, RZ, Pt0
//4: 4B 3E 4F 5G 6H
    [B------:R-:W-:-:S04]    IMAD.WIDE.U32 Rt4, Pt0, RFirst4, RSecond1, Rt4
    [B------:R-:W-:-:S04]    IMAD.WIDE.U32.X Rt0, Pt0, RFirst3, RSecond4, Rt0, Pt0
    [B------:R-:W-:-:S04]    IMAD.WIDE.U32.X Rt2, Pt0, RFirst4, RSecond5, Rt2, Pt0
    [B------:R-:W-:-:S04]    IMAD.WIDE.U32.X Rt6, Pt0, RFirst5, RSecond6, Rt6, Pt0
    [B------:R-:W-:-:S02]    IMAD.WIDE.U32.X Rt8, RFirst6, RSecond7, RZ, Pt0
//4: 5A 4D 5E 6F 7G, get C7(store in Pt4)
    [B------:R-:W-:-:S04]    IMAD.WIDE.U32 Rt4, Pt0, RFirst5, RSecond0, Rt4
    [B------:R-:W-:-:S02]    IMAD.WIDE.U32.X Rt0, Pt0, RFirst4, RSecond3, Rt0, Pt0
    [B------:R-:W-:-:S01]    IADD3.X Ro5, Pt3, Ro5, Rt4, RZ, Pt3, !PT //build res
    [B------:R-:W-:-:S02]    IMAD.WIDE.U32.X Rt2, Pt0, RFirst5, RSecond4, Rt2, Pt0
    [B------:R-:W-:-:S01]    IADD3.X Ro6, Pt3, Ro6, Rt5, RZ, Pt3, !PT //build res
    [B------:R-:W-:-:S04]    IMAD.WIDE.U32.X Rt6, Pt0, RFirst6, RSecond5, Rt6, Pt0
    [B------:R-:W-:-:S02]    IMAD.WIDE.U32.X Rt8, Pt4, RFirst7, RSecond6, Rt8, Pt0
//6: 5C 6D 7E, get C6
    [B------:R-:W-:-:S03]    IMAD.WIDE.U32 Rt0, Pt0, RFirst5, RSecond2, Rt0
    [B------:R-:W-:-:S01]    IADD3.X Rt19, PT, PT, Rt19, RZ, RZ, Pt1, Pt4 //now use C3 and C7 and forget
    [B------:R-:W-:-:S04]    IMAD.WIDE.U32.X Rt2, Pt0, RFirst6, RSecond3, Rt2, Pt0
    [B------:R-:W-:-:S02]    IMAD.WIDE.U32.X Rt6, Pt1, RFirst7, RSecond4, Rt6, Pt0 //C6 to Pt1
//6: 6B 7C, get C5
    [B------:R-:W-:-:S04]    IMAD.WIDE.U32 Rt0, Pt0, RFirst6, RSecond1, Rt0

    [B------:R-:W-:-:S02]    IMAD.WIDE.U32.X Rt2, Pt2, RFirst7, RSecond2, Rt2, Pt0
//6: 7A, get C4
    [B------:R-:W-:-:S02]    IMAD.WIDE.U32 Rt0, Pt4, RFirst7, RSecond0, Rt0
    [B------:R-:W-:-:S01]    IADD3.X Rt4, RZ, RZ, RZ, Pt2, !PT //C5 to Rt4
    [B------:R-:W-:-:S04]    IADD3.X Ro7, Pt3, Ro7, Rt0, RZ, Pt3, !PT //build res
    [B------:R-:W-:-:S04]    IADD3.X Rt12, Pt3, Rt12, Rt1, RZ, Pt3, !PT //build res
    [B------:R-:W-:-:S01]    IADD3.X Rt13, Pt3, Pt4, Rt13, Rt2, RZ, Pt3, Pt4 //build res
    [B------:R-:W-:-:S02]    IMAD.WIDE.U32 Rt0, Rt12, 0x3D1, RZ
    [B------:R-:W-:-:S01]    IADD3.X Rt14, Pt3, Pt4, Rt14, Rt3, Rt10, Pt3, Pt4 //build res
    [B------:R-:W-:-:S01]    IADD3.X Rt10, RZ, RZ, RZ, Pt1, !PT //C6 to Rt10
    [B------:R-:W-:-:S02]    IMAD.WIDE.U32.X Rt2, Pt2, Rt13, 0x3D1, Rt12, !PT
    [B------:R-:W-:-:S01]    IADD3.X Rt15, Pt3, Pt4, Rt15, Rt6, Rt4, Pt3, Pt4 //build res
    [B------:R-:W-:-:S01]    IADD3.X Ro0, Pt0, Pt1, Ro0, Rt0, RZ, !PT, !PT
    [B------:R-:W-:-:S01]    IADD3.X Rt16, Pt3, Pt4, Rt16, Rt7, Rt11, Pt3, Pt4 //build res
    [B------:R-:W-:-:S02]    IMAD.WIDE.U32 Rt4, Rt14, 0x3D1, RZ
    [B------:R-:W-:-:S01]    IADD3.X Rt17, Pt3, Pt4, Rt17, Rt8, Rt10, Pt3, Pt4 //build res
    [B------:R-:W-:-:S01]    IADD3.X Ro1, Pt0, Pt1, Rt1, Rt2, Ro1, Pt0, Pt1
    [B------:R-:W-:-:S01]    IADD3.X Rt18, Pt3, Pt4, Rt18, Rt9, RZ, Pt3, Pt4 //build res
    [B------:R-:W-:-:S02]    IMAD.WIDE.U32.X Rt6, Pt2, Rt15, 0x3D1, Rt14, Pt2
    [B------:R-:W-:-:S01]    IADD3.X Rt19, Rt19, RZ, RZ, Pt3, Pt4 //build res
// reduction
    [B------:R-:W-:-:S01]    IADD3.X Ro2, Pt0, Pt1, Rt3, Rt4, Ro2, Pt0, Pt1
    [B------:R-:W-:-:S02]    IMAD.WIDE.U32 Rt8, Rt16, 0x3D1, RZ
    [B------:R-:W-:-:S01]    IADD3.X Ro3, Pt0, Pt1, Rt5, Rt6, Ro3, Pt0, Pt1
    [B------:R-:W-:-:S02]    IMAD.WIDE.U32.X Rt4, Pt2, Rt17, 0x3D1, Rt16, Pt2
    [B------:R-:W-:-:S02]    IMAD.WIDE.U32 Rt0, Rt18, 0x3D1, RZ
    [B------:R-:W-:-:S01]    IADD3.X Ro4, Pt0, Pt1, Rt7, Rt8, Ro4, Pt0, Pt1
    [B------:R-:W-:-:S02]    IMAD.WIDE.U32.X Rt2, Pt2, Rt19, 0x3D1, Rt18, Pt2
    [B------:R-:W-:-:S04]    IADD3.X Ro5, Pt0, Pt1, Rt9, Rt4, Ro5, Pt0, Pt1
    [B------:R-:W-:-:S04]    IADD3.X Ro6, Pt0, Pt1, Rt5, Rt0, Ro6, Pt0, Pt1
    [B------:R-:W-:-:S04]    IADD3.X Ro7, Pt0, Pt1, Rt1, Rt2, Ro7, Pt0, Pt1
//Rt8..Rt9 is "tmp[4] in cpu code, Rt8 <= 0x3d1 (checked P-1*P-1), Rt9 <= 1
    [B------:R-:W-:-:S04]    IADD3.X Rt8, Pt0, PT, Rt3, RZ, RZ, Pt0, Pt1
    [B------:R-:W-:-:S01]    IADD3.X Rt9, Pt0, Pt1, RZ, RZ, RZ, Pt0, Pt2
////
    [B------:R-:W-:-:S02]    IMAD.WIDE.U32.X Ro0, Pt3, Rt8, 0x3D1, Ro0, !PT
    [B------:R-:W-:-:S03]    IMAD.WIDE.U32 Rt2, Rt9, 0x3D1, Rt8 //no carry occurs!
    [B------:R-:W-:-:S04]    IADD3.X Ro1, Pt0, PT, Ro1, Rt2, RZ, !PT, !PT
    [B------:R-:W-:-:S04]    IADD3.X Ro2, Pt0, Pt1, Rt3, Ro2, RZ, Pt0, Pt3
    [B------:R-:W-:-:S04]    IADD3.X Ro3, Pt0, PT, Ro3, RZ, RZ, Pt0, Pt1
    [B------:R-:W-:-:S04]    IADD3.X Ro4, Pt0, PT, Ro4, RZ, RZ, Pt0, !PT
    [B------:R-:W-:-:S04]    IADD3.X Ro5, Pt0, PT, Ro5, RZ, RZ, Pt0, !PT
    [B------:R-:W-:-:S04]    IADD3.X Ro6, Pt0, PT, Ro6, RZ, RZ, Pt0, !PT
    [B------:R-:W-:-:S01]    IADD3.X Ro7, PT, PT, Ro7, RZ, RZ, Pt0, !PT
}