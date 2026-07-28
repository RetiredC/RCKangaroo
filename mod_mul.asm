//(c) 2026, Retired Coder(RC)

//time for 4090: about 120G/sec
FUNCTION MulMod256()
{ //Ri_cnt=8+8, Ro_cnt=8, Rt_cnt=20, P=[0..4]
////start MulMod256
//0: 0A 0C
    [B------:R-:W-:-:S01]    IMAD.WIDE.U32 Ro0, RFirst0.reuse, RSecond0, RZ
    [B------:R-:W-:-:S01]    IMAD.WIDE.U32 Ro2, RFirst0, RSecond2, RZ
//2: 1B 0E
    [B------:R-:W-:-:S04]    IMAD.WIDE.U32 Ro2, Pt4, RFirst1.reuse, RSecond1, Ro2
    [B------:R-:W-:-:S01]    IMAD.WIDE.U32.X Ro4, RFirst0, RSecond4, RZ, Pt4
//2: 2A 1D 0G
    [B------:R-:W-:-:S04]    IMAD.WIDE.U32 Ro2, Pt4, RFirst2, RSecond0, Ro2
    [B------:R-:W-:-:S04]    IMAD.WIDE.U32.X Ro4, Pt4, RFirst1.reuse, RSecond3, Ro4, Pt4
    [B------:R-:W-:-:S01]    IMAD.WIDE.U32.X Ro6, RFirst0, RSecond6, RZ, Pt4
//4: 2C 1F 1H
    [B------:R-:W-:-:S04]    IMAD.WIDE.U32 Ro4, Pt4, RFirst2, RSecond2, Ro4
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
    [B------:R-:W-:-:S04]    IMAD.WIDE.U32 Ro6, Pt4, RFirst4.reuse, RSecond2, Ro6
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
    [B------:R-:W-:-:S02]    IMAD.WIDE.U32 Rt12, Pt1, RFirst7, RSecond1, Rt12
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
    [B------:R-:W-:-:S04]    IADD3.X Ro7, PT, PT, Ro7, RZ, RZ, Pt0, !PT
//now if P(carry) is 1 then we need Sub(256k1_P), chance is about 1/2^129 so just ignore it
////end MulMod256
}


FUNCTION MadMod256()
{ //Ri_cnt=8+8, Ro_cnt=8, Rt_cnt=20, P=[0..4]
////start MulMod256
//0: 0A 0C
    [B------:R-:W-:-:S01]    IMAD.WIDE.U32 Ro0, Pt0, RFirst0.reuse, RSecond0, Radd0 //Radd0
    [B------:R-:W-:-:S01]    IMAD.WIDE.U32 Ro2, Pt1, RFirst0, RSecond2, Radd2 //Radd2
//2: 1B 0E
    [B------:R-:W-:-:S04]    IMAD.WIDE.U32 Ro2, Pt4, RFirst1.reuse, RSecond1, Ro2
    [B------:R-:W-:-:S01]    IMAD.WIDE.U32.X Ro4, Pt2, RFirst0, RSecond4, Radd4, Pt4 //Radd4
//2: 2A 1D 0G
    [B------:R-:W-:-:S04]    IMAD.WIDE.U32 Ro2, Pt4, RFirst2, RSecond0, Ro2, Pt0 //cr0
    [B------:R-:W-:-:S04]    IMAD.WIDE.U32.X Ro4, Pt4, RFirst1.reuse, RSecond3, Ro4, Pt4
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
    [B------:R-:W-:-:S04]    IADD3.X Ro7, PT, PT, Ro7, RZ, RZ, Pt0, !PT
//now if P(carry) is 1 then we need Sub(256k1_P), chance is about 1/2^129 so just ignore it
////end MulMod256
}


FUNCTION SqrMod256(First=Ri0)
{ //Ri_cnt=8, Ro_cnt=8, Ro_tmp=26, P=[0..5]
////start SqrMod256
//2: 0A 0C 1B
    [B------:R-:W-:-:S01]    IMAD.WIDE.U32 Rt0, First0, First2, RZ
    [B------:R-:W-:-:S01]    IMAD.WIDE.U32 Ro2, First1, First1, RZ
//4: OE 1D 2C
    [B------:R-:W-:-:S01]    IMAD.WIDE.U32 Rt2, First0, First4, RZ
    [B------:R-:W-:-:S01]    IADD3.X Ro2, Pt0, Pt1, Ro2, Rt0, Rt0, !PT, !PT

    [B------:R-:W-:-:S01]    IMAD.WIDE.U32 Ro4, First2, First2, RZ
    [B------:R-:W-:-:S01]    IADD3.X Ro3, Pt0, Pt1, Ro3, Rt1, Rt1, Pt0, Pt1
    [B------:R-:W-:-:S01]    IMAD.WIDE.U32 Rt4, First1, First3, RZ
    [B------:R-:W-:-:S01]    IADD3.X Ro4, Pt0, Pt1, Ro4, Rt2, Rt2, Pt0, Pt1
    [B------:R-:W-:-:S01]    IMAD.WIDE.U32 Ro0, First0, First0, RZ
    [B------:R-:W-:-:S01]    IADD3.X Ro4, Pt2, Pt3, Ro4, Rt4, Rt4, !PT, !PT
//6: 0G 1F 2E 3D
    [B------:R-:W-:-:S02]    IMAD.WIDE.U32 Rt6, First0, First6, RZ
    [B------:R-:W-:-:S02]    IADD3.X Ro5, Pt0, Pt1, Ro5, Rt3, Rt3, Pt0, Pt1
    [B------:R-:W-:-:S02]    IMAD.WIDE.U32 Ro6, First3, First3, RZ
    [B------:R-:W-:-:S02]    IADD3.X Ro5, Pt2, Pt3, Ro5, Rt5, Rt5, Pt2, Pt3
    [B------:R-:W-:-:S02]    IMAD.WIDE.U32 Rt8,  First1, First5, RZ
    [B------:R-:W-:-:S02]    IADD3.X Ro6, Pt0, Pt1, Ro6, Rt6, Rt6, Pt0, Pt1
    [B------:R-:W-:-:S02]    IMAD.WIDE.U32 Rt10,  First2, First4, RZ
    [B------:R-:W-:-:S02]    IADD3.X Ro6, Pt2, Pt3, Ro6, Rt8, Rt8, Pt2, Pt3
//8: 1H 2G 3F 4E
    [B------:R-:W-:-:S02]    IMAD.WIDE.U32 Rt12, First1, First7, RZ
    [B------:R-:W-:-:S02]    IADD3.X Ro6, Pt4, Pt5, Ro6, Rt10, Rt10, !PT, !PT
    [B------:R-:W-:-:S02]    IMAD.WIDE.U32 Rt14, First2, First6, RZ
    [B------:R-:W-:-:S02]    IADD3.X Ro7, Pt0, Pt1, Ro7, Rt7, Rt7, Pt0, Pt1
    [B------:R-:W-:-:S02]    IMAD.WIDE.U32 Rt0, First3, First5, RZ
    [B------:R-:W-:-:S02]    IADD3.X Ro7, Pt2, Pt3, Ro7, Rt9, Rt9, Pt2, Pt3
    [B------:R-:W-:-:S02]    IMAD.WIDE.U32 Rt18, First4, First4, RZ
    [B------:R-:W-:-:S03]    IADD3.X Ro7, Pt4, Pt5, Ro7, Rt11, Rt11, Pt4, Pt5
//10: 3H 4G 5F
    [B------:R-:W-:-:S01]    IMAD.WIDE.U32 Rt10, First3, First7, RZ
    [B------:R-:W-:-:S01]    IADD3.X Rt18, Pt0, Pt1, Rt18, Rt12, Rt12, Pt0, Pt1
    [B------:R-:W-:-:S01]    IMAD.WIDE.U32 Rt6, First4, First6, RZ
    [B------:R-:W-:-:S01]    IADD3.X Rt18, Pt2, Pt3, Rt18, Rt14, Rt14, Pt2, Pt3
//12: 5H 6H 7H
    [B------:R-:W-:-:S01]    IMAD.WIDE.U32 Rt8, First5, First7, RZ
    [B------:R-:W-:-:S01]    IADD3.X Rt18, Pt4, Pt5, Rt18, Rt0, Rt0, Pt4, Pt5
    [B------:R-:W-:-:S01]    IMAD.WIDE.U32 Rt24, First7, First7, RZ
    [B------:R-:W-:-:S01]    IADD3.X Rt19, Pt4, Pt5, Rt19, Rt1, Rt1, Pt4, Pt5  //Pt5 = C1
    [B------:R-:W-:-:S01]    IMAD.WIDE.U32 Rt0, First0, First1, RZ
    [B------:R-:W-:-:S01]    IADD3.X Rt19, Pt0, Pt1, Rt19, Rt13, Rt13, Pt0, Pt1
    [B------:R-:W-:-:S01]    IADD3.X Rt14, RZ, RZ, RZ, Pt5, !PT //C1 to Rt14
    [B------:R-:W-:-:S01]    IMAD.WIDE.U32.X Rt20, PT, First5, First5, RZ, Pt4 //take one carry to imad
    [B------:R-:W-:-:S01]    IMAD.WIDE.U32 Rt0, Pt5, First1, First0, Rt0
    [B------:R-:W-:-:S01]    IADD3.X Rt19, Pt2, Pt3, Rt19, Rt15, Rt15, Pt2, Pt3
    [B------:R-:W-:-:S01]    IADD3.X Rt20, Pt4, Pt1, Rt20, Rt10, Rt10, Pt0, Pt1
//10: add
    [B------:R-:W-:-:S01]    IMAD.WIDE.U32.X Rt2, First0, First3, RZ, Pt5
    [B------:R-:W-:-:S01]    IADD3.X Rt20, Pt2, Pt3, Rt20, Rt6, Rt6, Pt2, Pt3
    [B------:R-:W-:-:S01]    IMAD.WIDE.U32 Rt2, Pt5, First1, First2, Rt2
    [B------:R-:W-:-:S02]    IADD3.X Rt21, Pt4, Pt1, Rt21, Rt11, Rt11, Pt4, Pt1
    [B------:R-:W-:-:S01]    IMAD.WIDE.U32.X Rt4, First0, First5, RZ, Pt5
    [B------:R-:W-:-:S01]    IADD3.X Rt21, Pt2, Pt3, Rt21, Rt7, Rt7, Pt2, Pt3 //Pt3 = C2
    [B------:R-:W-:-:S01]    IMAD.WIDE.U32 Rt2, Pt0, First2, First1, Rt2
    [B------:R-:W-:-:S01]    IADD3.X Rt15, RZ, RZ, RZ, Pt3, !PT //C2 to Rt15
//12: add
    [B------:R-:W-:-:S01]    IMAD.WIDE.U32.X Rt22, PT, First6, First6, RZ, Pt2 //take one carry to imad
    [B------:R-:W-:-:S01]    IADD3 Ro1, Pt3, Ro1, Rt0, RZ //build res
    [B------:R-:W-:-:S01]    IADD3.X Rt22, Pt4, Pt1, Rt22, Rt8, Rt8, Pt4, Pt1
    [B------:R-:W-:-:S01]    IMAD.WIDE.U32.X Rt4, Pt0, First1, First4, Rt4, Pt0
    [B------:R-:W-:-:S02]    IADD3.X Rt23, Pt4, Pt1, Rt23, Rt9, Rt9, Pt4, Pt1  //Pt1 = C3
    [B------:R-:W-:-:S01]    IMAD.WIDE.U32.X Rt6, First0, First7, RZ, Pt0
    [B------:R-:W-:-:S01]    IADD3.X Rt24, Pt4, Pt1, Rt24, RZ, RZ, Pt4, Pt1 //process C3 and forget
    [B------:R-:W-:-:S01]    IMAD.WIDE.U32 Rt2, Pt0, First3, First0, Rt2
//14:
    [B------:R-:W-:-:S01]    IADD3.X Rt25, PT, PT, Rt25, RZ, RZ, Pt4, Pt1
    [B------:R-:W-:-:S01]    IADD3.X Ro2, Pt3, Ro2, Rt1, RZ, Pt3, !PT //build res
//2: 3A 2D 1G 2H
    [B------:R-:W-:-:S02]    IMAD.WIDE.U32.X Rt4, Pt0, First2, First3, Rt4, Pt0
    [B------:R-:W-:-:S01]    IADD3.X Ro3, Pt3, Ro3, Rt2, RZ, Pt3, !PT //build res
    [B------:R-:W-:-:S03]    IMAD.WIDE.U32.X Rt6, Pt0, First1, First6, Rt6, Pt0
    [B------:R-:W-:-:S01]    IADD3.X Ro4, Pt3, Ro4, Rt3, RZ, Pt3, !PT //build res
    [B------:R-:W-:-:S01]    IMAD.WIDE.U32.X Rt8, First2, First7, RZ, Pt0
//4: 3C 2F 3G 4H
    [B------:R-:W-:Y:S04]    IMAD.WIDE.U32 Rt4, Pt0, First3, First2, Rt4
    [B------:R-:W-:Y:S04]    IMAD.WIDE.U32.X Rt6, Pt0, First2, First5, Rt6, Pt0
    [B------:R-:W-:Y:S04]    IMAD.WIDE.U32.X Rt8, Pt0, First3, First6, Rt8, Pt0
    [B------:R-:W-:-:S01]    IMAD.WIDE.U32.X Rt10, First4, First7, RZ, Pt0
//4: 4B 3E 4F 5G 6H
    [B------:R-:W-:Y:S04]    IMAD.WIDE.U32 Rt4, Pt0, First4, First1, Rt4
    [B------:R-:W-:Y:S04]    IMAD.WIDE.U32.X Rt6, Pt0, First3, First4, Rt6, Pt0
    [B------:R-:W-:Y:S04]    IMAD.WIDE.U32.X Rt8, Pt0, First4, First5, Rt8, Pt0
    [B------:R-:W-:Y:S04]    IMAD.WIDE.U32.X Rt10, Pt0, First5, First6, Rt10, Pt0
    [B------:R-:W-:-:S01]    IMAD.WIDE.U32.X Rt12, First6, First7, RZ, Pt0
//4: 5A 4D 5E 6F 7G, get C7(store in Pt4)
    [B------:R-:W-:Y:S04]    IMAD.WIDE.U32 Rt4, Pt0, First5, First0, Rt4
    [B------:R-:W-:-:S02]    IMAD.WIDE.U32.X Rt6, Pt0, First4, First3, Rt6, Pt0
    [B------:R-:W-:-:S01]    IADD3.X Ro5, Pt3, Ro5, Rt4, RZ, Pt3, !PT //build res
    [B------:R-:W-:-:S02]    IMAD.WIDE.U32.X Rt8, Pt0, First5, First4, Rt8, Pt0
    [B------:R-:W-:-:S01]    IADD3.X Ro6, Pt3, Ro6, Rt5, RZ, Pt3, !PT //build res
    [B------:R-:W-:Y:S04]    IMAD.WIDE.U32.X Rt10, Pt0, First6, First5, Rt10, Pt0
    [B------:R-:W-:-:S01]    IMAD.WIDE.U32.X Rt12, Pt4, First7, First6, Rt12, Pt0
//6: 5C 6D 7E, get C6
    [B------:R-:W-:-:S03]    IMAD.WIDE.U32 Rt6, Pt0, First5, First2, Rt6
    [B------:R-:W-:-:S01]    IADD3.X Rt25, PT, PT, Rt25, RZ, RZ, !PT, Pt4 //process C7 and forget
    [B------:R-:W-:Y:S04]    IMAD.WIDE.U32.X Rt8, Pt0, First6, First3, Rt8, Pt0
    [B------:R-:W-:-:S01]    IMAD.WIDE.U32.X Rt10, Pt2, First7, First4, Rt10, Pt0 //C6 to Pt2
//6: 6B 7C, get C5
    [B------:R-:W-:-:S03]    IMAD.WIDE.U32 Rt6, Pt0, First6, First1, Rt6
    [B------:R-:W-:-:S01]    IADD3.X Rt17, RZ, RZ, RZ, Pt2, !PT //C6 to Rt17
    [B------:R-:W-:-:S01]    IMAD.WIDE.U32.X Rt8, Pt2, First7, First2, Rt8, Pt0
//6: 7A, get C4
    [B------:R-:W-:-:S01]    IMAD.WIDE.U32 Rt6, Pt4, First7, First0, Rt6
    [B------:R-:W-:-:S01]    IADD3.X Rt16, RZ, RZ, RZ, Pt2, !PT //C5 to Rt16
    [B------:R-:W-:Y:S04]    IADD3.X Ro7, Pt3, Ro7, Rt6, RZ, Pt3, !PT //build res
    [B------:R-:W-:Y:S04]    IADD3.X Rt18, Pt3, Rt18, Rt7, RZ, Pt3, !PT //build res
    [B------:R-:W-:-:S01]    IADD3.X Rt19, Pt3, Pt4, Rt19, Rt8, RZ, Pt3, Pt4 //build res
    [B------:R-:W-:-:S01]    IMAD.WIDE.U32 Rt0, Rt18, 0x3D1, RZ
    [B------:R-:W-:-:S01]    IADD3.X Rt20, Pt3, Pt4, Rt20, Rt9, Rt14, Pt3, Pt4 //build res
    [B------:R-:W-:-:S01]    IMAD.WIDE.U32.X Rt2, Pt2, Rt19, 0x3D1, Rt18, !PT
    [B------:R-:W-:-:S01]    IADD3.X Rt21, Pt3, Pt4, Rt21, Rt10, Rt16, Pt3, Pt4 //build res
    [B------:R-:W-:-:S01]    IADD3.X Ro0, Pt0, Pt1, Ro0, Rt0, RZ, !PT, !PT
    [B------:R-:W-:-:S01]    IADD3.X Rt22, Pt3, Pt4, Rt22, Rt11, Rt15, Pt3, Pt4 //build res
    [B------:R-:W-:-:S01]    IMAD.WIDE.U32 Rt4, Rt20, 0x3D1, RZ
    [B------:R-:W-:-:S01]    IADD3.X Rt23, Pt3, Pt4, Rt23, Rt12, Rt17, Pt3, Pt4 //build res
    [B------:R-:W-:-:S01]    IADD3.X Ro1, Pt0, Pt1, Rt1, Rt2, Ro1, Pt0, Pt1
    [B------:R-:W-:-:S01]    IADD3.X Rt24, Pt3, Pt4, Rt24, Rt13, RZ, Pt3, Pt4 //build res
    [B------:R-:W-:-:S01]    IMAD.WIDE.U32.X Rt6, Pt2, Rt21, 0x3D1, Rt20, Pt2
    [B------:R-:W-:-:S01]    IADD3.X Rt25, Rt25, RZ, RZ, Pt3, Pt4 //build res
///////////////////////////////////////////////// reduction
    [B------:R-:W-:-:S01]    IADD3.X Ro2, Pt0, Pt1, Rt3, Rt4, Ro2, Pt0, Pt1
    [B------:R-:W-:-:S01]    IMAD.WIDE.U32 Rt8, Rt22, 0x3D1, RZ
    [B------:R-:W-:-:S01]    IADD3.X Ro3, Pt0, Pt1, Rt5, Rt6, Ro3, Pt0, Pt1
    [B------:R-:W-:-:S01]    IMAD.WIDE.U32.X Rt10, Pt2, Rt23, 0x3D1, Rt22, Pt2
    [B------:R-:W-:-:S01]    IMAD.WIDE.U32 Rt12, Rt24, 0x3D1, RZ
    [B------:R-:W-:-:S01]    IADD3.X Ro4, Pt0, Pt1, Rt7, Rt8, Ro4, Pt0, Pt1
    [B------:R-:W-:-:S01]    IMAD.WIDE.U32.X Rt14, Pt2, Rt25, 0x3D1, Rt24, Pt2
    [B------:R-:W-:Y:S04]    IADD3.X Ro5, Pt0, Pt1, Rt9, Rt10, Ro5, Pt0, Pt1
    [B------:R-:W-:Y:S04]    IADD3.X Ro6, Pt0, Pt1, Rt11, Rt12, Ro6, Pt0, Pt1
    [B------:R-:W-:Y:S04]    IADD3.X Ro7, Pt0, Pt1, Rt13, Rt14, Ro7, Pt0, Pt1
//Rt8..Rt9 is "tmp[4] in cpu code, Rt8 <= 0x3d1 (checked P-1*P-1), Rt9 <= 1
    [B------:R-:W-:Y:S04]    IADD3.X Rt8, Pt0, PT, Rt15, RZ, RZ, Pt0, Pt1
    [B------:R-:W-:-:S01]    IADD3.X Rt9, Pt0, Pt1, RZ, RZ, RZ, Pt0, Pt2
/////////////////////////////
    [B------:R-:W-:-:S01]    IMAD.WIDE.U32.X Ro0, Pt3, Rt8, 0x3D1, Ro0, !PT
    [B------:R-:W-:-:S03]    IMAD.WIDE.U32 Rt2, Rt9, 0x3D1, Rt8 //no carry occurs!
    [B------:R-:W-:Y:S04]    IADD3.X Ro1, Pt0, PT, Ro1, Rt2, RZ, !PT, !PT
    [B------:R-:W-:Y:S04]    IADD3.X Ro2, Pt0, Pt1, Rt3, Ro2, RZ, Pt0, Pt3
    [B------:R-:W-:Y:S04]    IADD3.X Ro3, Pt0, PT, Ro3, RZ, RZ, Pt0, Pt1
    [B------:R-:W-:Y:S04]    IADD3.X Ro4, Pt0, PT, Ro4, RZ, RZ, Pt0, !PT
    [B------:R-:W-:Y:S04]    IADD3.X Ro5, Pt0, PT, Ro5, RZ, RZ, Pt0, !PT
    [B------:R-:W-:Y:S04]    IADD3.X Ro6, Pt0, PT, Ro6, RZ, RZ, Pt0, !PT
    [B------:R-:W-:Y:S04]    IADD3.X Ro7, PT, PT, Ro7, RZ, RZ, Pt0, !PT
//now if P(carry) is 1 then we need Sub(256k1_P), chance is about 1/2^129 so just ignore it

////end SqrMod256
}


FUNCTION SqrAddMod256(First=Ri0)
{ //Ri_cnt=8, Ro_cnt=8, Ro_tmp=26, P=[0..5]
////start SqrMod256
//2: 0A 0C 1B
    [B------:R-:W-:-:S01]    IMAD.WIDE.U32 Ro0, Pt4, First0, First0, Radd0 //Radd0, cr0=Pt4
    [B------:R-:W-:-:S01]    IMAD.WIDE.U32 Rt0, First0, First2, RZ
    [B------:R-:W-:-:S01]    IMAD.WIDE.U32.X Ro2, Pt4, First1, First1, Radd2, Pt4 //Radd2, cr2=Pt4
//4: OE 1D 2C
    [B------:R-:W-:-:S01]    IMAD.WIDE.U32 Rt2, First0, First4, RZ
    [B------:R-:W-:-:S01]    IADD3.X Ro2, Pt0, Pt1, Ro2, Rt0, Rt0, !PT, !PT
    [B------:R-:W-:-:S01]    IMAD.WIDE.U32.X Ro4, Pt4, First2, First2, Radd4, Pt4 //Radd4, cr4=Pt4     cr2(Pt4)
    [B------:R-:W-:-:S01]    IADD3.X Ro3, Pt0, Pt1, Ro3, Rt1, Rt1, Pt0, Pt1
    [B------:R-:W-:-:S01]    IMAD.WIDE.U32 Rt4, First1, First3, RZ
    [B------:R-:W-:-:S04]    IADD3.X Ro4, Pt0, Pt1, Ro4, Rt2, Rt2, Pt0, Pt1
    [B------:R-:W-:-:S01]    IADD3.X Ro4, Pt2, Pt3, Ro4, Rt4, Rt4, !PT, !PT
//6: 0G 1F 2E 3D
    [B------:R-:W-:-:S02]    IMAD.WIDE.U32 Rt6, First0, First6, RZ
    [B------:R-:W-:-:S02]    IADD3.X Ro5, Pt0, Pt1, Ro5, Rt3, Rt3, Pt0, Pt1
    [B------:R-:W-:-:S02]    IMAD.WIDE.U32.X Ro6, Pt4, First3, First3, Radd6, Pt4 //Radd6, cr6=Pt4    cr4(Pt4)
    [B------:R-:W-:-:S02]    IADD3.X Ro5, Pt2, Pt3, Ro5, Rt5, Rt5, Pt2, Pt3
    [B------:R-:W-:-:S02]    IMAD.WIDE.U32 Rt8,  First1, First5, RZ
    [B------:R-:W-:-:S02]    IADD3.X Ro6, Pt0, Pt1, Ro6, Rt6, Rt6, Pt0, Pt1
    [B------:R-:W-:-:S02]    IMAD.WIDE.U32 Rt10,  First2, First4, RZ
    [B------:R-:W-:-:S02]    IADD3.X Ro6, Pt2, Pt3, Ro6, Rt8, Rt8, Pt2, Pt3
//8: 1H 2G 3F 4E
    [B------:R-:W-:-:S01]    IMAD.WIDE.U32.X Rt18, Pt4, First4, First4, RNineTen, Pt4 //RNineTen, cr8=Pt4    cr6(Pt4)
    [B------:R-:W-:-:S01]    IMAD.WIDE.U32 Rt12, First1, First7, RZ
    [B------:R-:W-:-:S01]    IADD3.X RNineTen0, PT, PT, RZ, RZ, RZ, Pt4, !PT //spoils RNineTen0, RNineTen1 must be 0
    [B------:R-:W-:-:S02]    IADD3.X Ro6, Pt4, Pt5, Ro6, Rt10, Rt10, !PT, !PT
    [B------:R-:W-:-:S02]    IMAD.WIDE.U32 Rt14, First2, First6, RZ
    [B------:R-:W-:-:S02]    IADD3.X Ro7, Pt0, Pt1, Ro7, Rt7, Rt7, Pt0, Pt1
    [B------:R-:W-:-:S02]    IMAD.WIDE.U32 Rt0, First3, First5, RZ
    [B------:R-:W-:-:S04]    IADD3.X Ro7, Pt2, Pt3, Ro7, Rt9, Rt9, Pt2, Pt3
    [B------:R-:W-:-:S04]    IADD3.X Ro7, Pt4, Pt5, Ro7, Rt11, Rt11, Pt4, Pt5
//10: 3H 4G 5F
    [B------:R-:W-:-:S01]    IMAD.WIDE.U32 Rt10, First3, First7, RZ
    [B------:R-:W-:-:S01]    IADD3.X Rt18, Pt0, Pt1, Rt18, Rt12, Rt12, Pt0, Pt1
    [B------:R-:W-:-:S01]    IMAD.WIDE.U32 Rt6, First4, First6, RZ
    [B------:R-:W-:-:S01]    IADD3.X Rt18, Pt2, Pt3, Rt18, Rt14, Rt14, Pt2, Pt3
//12: 5H 6H 7H
    [B------:R-:W-:-:S01]    IMAD.WIDE.U32 Rt8, First5, First7, RZ
    [B------:R-:W-:-:S01]    IADD3.X Rt18, Pt4, Pt5, Rt18, Rt0, Rt0, Pt4, Pt5
    [B------:R-:W-:-:S01]    IMAD.WIDE.U32 Rt24, First7, First7, RZ
    [B------:R-:W-:-:S01]    IADD3.X Rt19, Pt4, Pt5, Rt19, Rt1, Rt1, Pt4, Pt5  //Pt5 = C1
    [B------:R-:W-:-:S01]    IMAD.WIDE.U32 Rt0, First0, First1, RZ
    [B------:R-:W-:-:S01]    IADD3.X Rt19, Pt0, Pt1, Rt19, Rt13, Rt13, Pt0, Pt1
    [B------:R-:W-:-:S01]    IADD3.X Rt14, RZ, RZ, RZ, Pt5, !PT //C1 to Rt14
    [B------:R-:W-:-:S01]    IMAD.WIDE.U32.X Rt20, PT, First5, First5, RNineTen, Pt4 //take one carry to imad
    [B------:R-:W-:-:S01]    IMAD.WIDE.U32 Rt0, Pt5, First1, First0, Rt0
    [B------:R-:W-:-:S01]    IADD3.X Rt19, Pt2, Pt3, Rt19, Rt15, Rt15, Pt2, Pt3
    [B------:R-:W-:-:S01]    IADD3.X Rt20, Pt4, Pt1, Rt20, Rt10, Rt10, Pt0, Pt1
//10: add
    [B------:R-:W-:-:S01]    IMAD.WIDE.U32.X Rt2, First0, First3, RZ, Pt5
    [B------:R-:W-:-:S01]    IADD3.X Rt20, Pt2, Pt3, Rt20, Rt6, Rt6, Pt2, Pt3
    [B------:R-:W-:-:S01]    IMAD.WIDE.U32 Rt2, Pt5, First1, First2, Rt2
    [B------:R-:W-:-:S02]    IADD3.X Rt21, Pt4, Pt1, Rt21, Rt11, Rt11, Pt4, Pt1
    [B------:R-:W-:-:S01]    IMAD.WIDE.U32.X Rt4, First0, First5, RZ, Pt5
    [B------:R-:W-:-:S01]    IADD3.X Rt21, Pt2, Pt3, Rt21, Rt7, Rt7, Pt2, Pt3 //Pt3 = C2
    [B------:R-:W-:-:S01]    IMAD.WIDE.U32 Rt2, Pt0, First2, First1, Rt2
    [B------:R-:W-:-:S01]    IADD3.X Rt15, RZ, RZ, RZ, Pt3, !PT //C2 to Rt15
//12: add
    [B------:R-:W-:-:S01]    IMAD.WIDE.U32.X Rt22, PT, First6, First6, RZ, Pt2 //take one carry to imad
    [B------:R-:W-:-:S01]    IADD3 Ro1, Pt3, Ro1, Rt0, RZ //build res
    [B------:R-:W-:-:S01]    IADD3.X Rt22, Pt4, Pt1, Rt22, Rt8, Rt8, Pt4, Pt1
    [B------:R-:W-:-:S01]    IMAD.WIDE.U32.X Rt4, Pt0, First1, First4, Rt4, Pt0
    [B------:R-:W-:-:S02]    IADD3.X Rt23, Pt4, Pt1, Rt23, Rt9, Rt9, Pt4, Pt1  //Pt1 = C3
    [B------:R-:W-:-:S01]    IMAD.WIDE.U32.X Rt6, First0, First7, RZ, Pt0
    [B------:R-:W-:-:S01]    IADD3.X Rt24, Pt4, Pt1, Rt24, RZ, RZ, Pt4, Pt1 //process C3 and forget
    [B------:R-:W-:-:S01]    IMAD.WIDE.U32 Rt2, Pt0, First3, First0, Rt2
//14:
    [B------:R-:W-:-:S01]    IADD3.X Rt25, PT, PT, Rt25, RZ, RZ, Pt4, Pt1
    [B------:R-:W-:-:S01]    IADD3.X Ro2, Pt3, Ro2, Rt1, RZ, Pt3, !PT //build res
//2: 3A 2D 1G 2H
    [B------:R-:W-:-:S02]    IMAD.WIDE.U32.X Rt4, Pt0, First2, First3, Rt4, Pt0
    [B------:R-:W-:-:S01]    IADD3.X Ro3, Pt3, Ro3, Rt2, RZ, Pt3, !PT //build res
    [B------:R-:W-:-:S03]    IMAD.WIDE.U32.X Rt6, Pt0, First1, First6, Rt6, Pt0
    [B------:R-:W-:-:S01]    IADD3.X Ro4, Pt3, Ro4, Rt3, RZ, Pt3, !PT //build res
    [B------:R-:W-:-:S01]    IMAD.WIDE.U32.X Rt8, First2, First7, RZ, Pt0
//4: 3C 2F 3G 4H
    [B------:R-:W-:Y:S04]    IMAD.WIDE.U32 Rt4, Pt0, First3, First2, Rt4
    [B------:R-:W-:Y:S04]    IMAD.WIDE.U32.X Rt6, Pt0, First2, First5, Rt6, Pt0
    [B------:R-:W-:Y:S04]    IMAD.WIDE.U32.X Rt8, Pt0, First3, First6, Rt8, Pt0
    [B------:R-:W-:-:S01]    IMAD.WIDE.U32.X Rt10, First4, First7, RZ, Pt0
//4: 4B 3E 4F 5G 6H
    [B------:R-:W-:Y:S04]    IMAD.WIDE.U32 Rt4, Pt0, First4, First1, Rt4
    [B------:R-:W-:Y:S04]    IMAD.WIDE.U32.X Rt6, Pt0, First3, First4, Rt6, Pt0
    [B------:R-:W-:Y:S04]    IMAD.WIDE.U32.X Rt8, Pt0, First4, First5, Rt8, Pt0
    [B------:R-:W-:Y:S04]    IMAD.WIDE.U32.X Rt10, Pt0, First5, First6, Rt10, Pt0
    [B------:R-:W-:-:S01]    IMAD.WIDE.U32.X Rt12, First6, First7, RZ, Pt0
//4: 5A 4D 5E 6F 7G, get C7(store in Pt4)
    [B------:R-:W-:Y:S04]    IMAD.WIDE.U32 Rt4, Pt0, First5, First0, Rt4
    [B------:R-:W-:-:S02]    IMAD.WIDE.U32.X Rt6, Pt0, First4, First3, Rt6, Pt0
    [B------:R-:W-:-:S01]    IADD3.X Ro5, Pt3, Ro5, Rt4, RZ, Pt3, !PT //build res
    [B------:R-:W-:-:S02]    IMAD.WIDE.U32.X Rt8, Pt0, First5, First4, Rt8, Pt0
    [B------:R-:W-:-:S01]    IADD3.X Ro6, Pt3, Ro6, Rt5, RZ, Pt3, !PT //build res
    [B------:R-:W-:Y:S04]    IMAD.WIDE.U32.X Rt10, Pt0, First6, First5, Rt10, Pt0
    [B------:R-:W-:-:S01]    IMAD.WIDE.U32.X Rt12, Pt4, First7, First6, Rt12, Pt0
//6: 5C 6D 7E, get C6
    [B------:R-:W-:-:S03]    IMAD.WIDE.U32 Rt6, Pt0, First5, First2, Rt6
    [B------:R-:W-:-:S01]    IADD3.X Rt25, PT, PT, Rt25, RZ, RZ, !PT, Pt4 //process C7 and forget
    [B------:R-:W-:Y:S04]    IMAD.WIDE.U32.X Rt8, Pt0, First6, First3, Rt8, Pt0
    [B------:R-:W-:-:S01]    IMAD.WIDE.U32.X Rt10, Pt2, First7, First4, Rt10, Pt0 //C6 to Pt2
//6: 6B 7C, get C5
    [B------:R-:W-:-:S03]    IMAD.WIDE.U32 Rt6, Pt0, First6, First1, Rt6
    [B------:R-:W-:-:S01]    IADD3.X Rt17, RZ, RZ, RZ, Pt2, !PT //C6 to Rt17
    [B------:R-:W-:-:S01]    IMAD.WIDE.U32.X Rt8, Pt2, First7, First2, Rt8, Pt0
//6: 7A, get C4
    [B------:R-:W-:-:S01]    IMAD.WIDE.U32 Rt6, Pt4, First7, First0, Rt6
    [B------:R-:W-:-:S01]    IADD3.X Rt16, RZ, RZ, RZ, Pt2, !PT //C5 to Rt16
    [B------:R-:W-:Y:S04]    IADD3.X Ro7, Pt3, Ro7, Rt6, RZ, Pt3, !PT //build res
    [B------:R-:W-:Y:S04]    IADD3.X Rt18, Pt3, Rt18, Rt7, RZ, Pt3, !PT //build res
    [B------:R-:W-:-:S01]    IADD3.X Rt19, Pt3, Pt4, Rt19, Rt8, RZ, Pt3, Pt4 //build res
    [B------:R-:W-:-:S01]    IMAD.WIDE.U32 Rt0, Rt18, 0x3D1, RZ
    [B------:R-:W-:-:S01]    IADD3.X Rt20, Pt3, Pt4, Rt20, Rt9, Rt14, Pt3, Pt4 //build res
    [B------:R-:W-:-:S01]    IMAD.WIDE.U32.X Rt2, Pt2, Rt19, 0x3D1, Rt18, !PT
    [B------:R-:W-:-:S01]    IADD3.X Rt21, Pt3, Pt4, Rt21, Rt10, Rt16, Pt3, Pt4 //build res
    [B------:R-:W-:-:S01]    IADD3.X Ro0, Pt0, Pt1, Ro0, Rt0, RZ, !PT, !PT
    [B------:R-:W-:-:S01]    IADD3.X Rt22, Pt3, Pt4, Rt22, Rt11, Rt15, Pt3, Pt4 //build res
    [B------:R-:W-:-:S01]    IMAD.WIDE.U32 Rt4, Rt20, 0x3D1, RZ
    [B------:R-:W-:-:S01]    IADD3.X Rt23, Pt3, Pt4, Rt23, Rt12, Rt17, Pt3, Pt4 //build res
    [B------:R-:W-:-:S01]    IADD3.X Ro1, Pt0, Pt1, Rt1, Rt2, Ro1, Pt0, Pt1
    [B------:R-:W-:-:S01]    IADD3.X Rt24, Pt3, Pt4, Rt24, Rt13, RZ, Pt3, Pt4 //build res
    [B------:R-:W-:-:S01]    IMAD.WIDE.U32.X Rt6, Pt2, Rt21, 0x3D1, Rt20, Pt2
    [B------:R-:W-:-:S01]    IADD3.X Rt25, Rt25, RZ, RZ, Pt3, Pt4 //build res
///////////////////////////////////////////////// reduction
    [B------:R-:W-:-:S01]    IADD3.X Ro2, Pt0, Pt1, Rt3, Rt4, Ro2, Pt0, Pt1
    [B------:R-:W-:-:S01]    IMAD.WIDE.U32 Rt8, Rt22, 0x3D1, RZ
    [B------:R-:W-:-:S01]    IADD3.X Ro3, Pt0, Pt1, Rt5, Rt6, Ro3, Pt0, Pt1
    [B------:R-:W-:-:S01]    IMAD.WIDE.U32.X Rt10, Pt2, Rt23, 0x3D1, Rt22, Pt2
    [B------:R-:W-:-:S01]    IMAD.WIDE.U32 Rt12, Rt24, 0x3D1, RZ
    [B------:R-:W-:-:S01]    IADD3.X Ro4, Pt0, Pt1, Rt7, Rt8, Ro4, Pt0, Pt1
    [B------:R-:W-:-:S01]    IMAD.WIDE.U32.X Rt14, Pt2, Rt25, 0x3D1, Rt24, Pt2
    [B------:R-:W-:Y:S04]    IADD3.X Ro5, Pt0, Pt1, Rt9, Rt10, Ro5, Pt0, Pt1
    [B------:R-:W-:Y:S04]    IADD3.X Ro6, Pt0, Pt1, Rt11, Rt12, Ro6, Pt0, Pt1
    [B------:R-:W-:Y:S04]    IADD3.X Ro7, Pt0, Pt1, Rt13, Rt14, Ro7, Pt0, Pt1
//Rt8..Rt9 is "tmp[4] in cpu code, Rt8 <= 0x3d1 (checked P-1*P-1), Rt9 <= 1
    [B------:R-:W-:Y:S04]    IADD3.X Rt8, Pt0, PT, Rt15, RZ, RZ, Pt0, Pt1
    [B------:R-:W-:-:S01]    IADD3.X Rt9, Pt0, Pt1, RZ, RZ, RZ, Pt0, Pt2
/////////////////////////////
    [B------:R-:W-:-:S01]    IMAD.WIDE.U32.X Ro0, Pt3, Rt8, 0x3D1, Ro0, !PT
    [B------:R-:W-:-:S03]    IMAD.WIDE.U32 Rt2, Rt9, 0x3D1, Rt8 //no carry occurs!
    [B------:R-:W-:Y:S04]    IADD3.X Ro1, Pt0, PT, Ro1, Rt2, RZ, !PT, !PT
    [B------:R-:W-:Y:S04]    IADD3.X Ro2, Pt0, Pt1, Rt3, Ro2, RZ, Pt0, Pt3
    [B------:R-:W-:Y:S04]    IADD3.X Ro3, Pt0, PT, Ro3, RZ, RZ, Pt0, Pt1
    [B------:R-:W-:Y:S04]    IADD3.X Ro4, Pt0, PT, Ro4, RZ, RZ, Pt0, !PT
    [B------:R-:W-:Y:S04]    IADD3.X Ro5, Pt0, PT, Ro5, RZ, RZ, Pt0, !PT
    [B------:R-:W-:Y:S04]    IADD3.X Ro6, Pt0, PT, Ro6, RZ, RZ, Pt0, !PT
    [B------:R-:W-:Y:S05]    IADD3.X Ro7, PT, PT, Ro7, RZ, RZ, Pt0, !PT
//now if P(carry) is 1 then we need Sub(256k1_P), chance is about 1/2^129 so just ignore it

////end SqrMod256
}