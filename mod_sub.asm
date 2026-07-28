//(c) 2026, Retired Coder(RC)

FUNCTION SubMod256()
{ //Ri_cnt=8+8, Ro_cnt=8, Pt=[0..1]
    [B------:R-:W-:-:S04]    IADD3.X Ro0, Pt0, PT, RFirst0, ~RSecond0, RZ, !PT, PT
    [B------:R-:W-:-:S04]    IADD3.X Ro1, Pt0, PT, RFirst1, ~RSecond1, RZ, Pt0, !PT
    [B------:R-:W-:-:S04]    IADD3.X Ro2, Pt0, PT, RFirst2, ~RSecond2, RZ, Pt0, !PT
    [B------:R-:W-:-:S04]    IADD3.X Ro3, Pt0, PT, RFirst3, ~RSecond3, RZ, Pt0, !PT
    [B------:R-:W-:-:S04]    IADD3.X Ro4, Pt0, PT, RFirst4, ~RSecond4, RZ, Pt0, !PT
    [B------:R-:W-:-:S04]    IADD3.X Ro5, Pt0, PT, RFirst5, ~RSecond5, RZ, Pt0, !PT
    [B------:R-:W-:-:S04]    IADD3.X Ro6, Pt0, PT, RFirst6, ~RSecond6, RZ, Pt0, !PT
    [B------:R-:W-:Y:S13]    IADD3.X Ro7, Pt0, PT, RFirst7, ~RSecond7, RZ, Pt0, !PT
//add P if <0
    [B------:R-:W-:-:S04] @!Pt0 IADD3.X Ro0, Pt1, PT, Ro0, 0xFFFFFC2F, RZ, !PT, !PT
    [B------:R-:W-:-:S04] @!Pt0 IADD3.X Ro1, Pt1, PT, Ro1, 0xFFFFFFFE, RZ, Pt1, !PT
    [B------:R-:W-:-:S04] @!Pt0 IADD3.X Ro2, Pt1, PT, Ro2, 0xFFFFFFFF, RZ, Pt1, !PT
    [B------:R-:W-:-:S04] @!Pt0 IADD3.X Ro3, Pt1, PT, Ro3, 0xFFFFFFFF, RZ, Pt1, !PT
    [B------:R-:W-:-:S04] @!Pt0 IADD3.X Ro4, Pt1, PT, Ro4, 0xFFFFFFFF, RZ, Pt1, !PT
    [B------:R-:W-:-:S04] @!Pt0 IADD3.X Ro5, Pt1, PT, Ro5, 0xFFFFFFFF, RZ, Pt1, !PT
    [B------:R-:W-:-:S04] @!Pt0 IADD3.X Ro6, Pt1, PT, Ro6, 0xFFFFFFFF, RZ, Pt1, !PT
    [B------:R-:W-:-:S05] @!Pt0 IADD3.X Ro7, PT, PT, Ro7, 0xFFFFFFFF, RZ, Pt1, !PT
}

FUNCTION SubMod256_3()
{ //Ri_cnt=8+8+8, Ro_cnt=8, Rt_cnt=2, Pt=[0..2]
//reg bank conflict is possible
    [B------:R-:W-:-:S04]    IADD3.X Ro0, Pt0, Pt2, RFirst0, ~RSecond0, ~RThird0, PT, PT
    [B------:R-:W-:-:S04]    IADD3.X Ro1, Pt0, Pt2, RFirst1, ~RSecond1, ~RThird1, Pt0, Pt2
    [B------:R-:W-:-:S04]    IADD3.X Ro2, Pt0, Pt2, RFirst2, ~RSecond2, ~RThird2, Pt0, Pt2
    [B------:R-:W-:-:S04]    IADD3.X Ro3, Pt0, Pt2, RFirst3, ~RSecond3, ~RThird3, Pt0, Pt2
    [B------:R-:W-:-:S04]    IADD3.X Ro4, Pt0, Pt2, RFirst4, ~RSecond4, ~RThird4, Pt0, Pt2
    [B------:R-:W-:-:S04]    IADD3.X Ro5, Pt0, Pt2, RFirst5, ~RSecond5, ~RThird5, Pt0, Pt2
    [B------:R-:W-:-:S01]    IADD3.X Ro6, Pt0, Pt2, RFirst6, ~RSecond6, ~RThird6, Pt0, Pt2
    [B------:R-:W-:-:S01]    MOV Rt1, 0xFFFFFC2F
    [B------:R-:W-:Y:S09]    IADD3.X Ro7, Pt0, Pt2, RFirst7, ~RSecond7, ~RThird7, Pt0, Pt2
    [B------:R-:W-:-:S04]    SEL Rt0, Rt1, 0xFFFFF85E, Pt0
    [B------:R-:W-:-:S04] @!Pt2 IADD3.X Ro0, Pt1, PT, Ro0, Rt0, RZ, !PT, !PT
    [B------:R-:W-:-:S04] @!Pt2 IADD3.X Ro1, Pt1, PT, Ro1, 0xFFFFFFFD, RZ, Pt1, Pt0
    [B------:R-:W-:-:S04] @!Pt2 IADD3.X Ro2, Pt1, PT, Ro2, 0xFFFFFFFF, RZ, Pt1, !PT
    [B------:R-:W-:-:S04] @!Pt2 IADD3.X Ro3, Pt1, PT, Ro3, 0xFFFFFFFF, RZ, Pt1, !PT
    [B------:R-:W-:-:S04] @!Pt2 IADD3.X Ro4, Pt1, PT, Ro4, 0xFFFFFFFF, RZ, Pt1, !PT
    [B------:R-:W-:-:S04] @!Pt2 IADD3.X Ro5, Pt1, PT, Ro5, 0xFFFFFFFF, RZ, Pt1, !PT
    [B------:R-:W-:-:S04] @!Pt2 IADD3.X Ro6, Pt1, PT, Ro6, 0xFFFFFFFF, RZ, Pt1, !PT
    [B------:R-:W-:-:S05] @!Pt2 IADD3.X Ro7, PT,  PT, Ro7, 0xFFFFFFFF, RZ, Pt1, !PT
}

FUNCTION AddMod256()
{ //Ri_cnt=8+8, Ro_cnt=8, Pt=[0..1]
    [B------:R-:W-:-:S04]    IADD3.X Ro0, Pt0, PT, RFirst0, RSecond0, RZ, !PT, !PT
    [B------:R-:W-:-:S04]    IADD3.X Ro1, Pt0, PT, RFirst1, RSecond1, RZ, Pt0, !PT
    [B------:R-:W-:-:S04]    IADD3.X Ro2, Pt0, PT, RFirst2, RSecond2, RZ, Pt0, !PT
    [B------:R-:W-:-:S04]    IADD3.X Ro3, Pt0, PT, RFirst3, RSecond3, RZ, Pt0, !PT
    [B------:R-:W-:-:S04]    IADD3.X Ro4, Pt0, PT, RFirst4, RSecond4, RZ, Pt0, !PT
    [B------:R-:W-:-:S04]    IADD3.X Ro5, Pt0, PT, RFirst5, RSecond5, RZ, Pt0, !PT
    [B------:R-:W-:-:S04]    IADD3.X Ro6, Pt0, PT, RFirst6, RSecond6, RZ, Pt0, !PT
    [B------:R-:W-:Y:S13]    IADD3.X Ro7, Pt0, PT, RFirst7, RSecond7, RZ, Pt0, !PT
//sub P if carry
    [B------:R-:W-:-:S04] @Pt0 IADD3.X Ro0, Pt1, PT, Ro0, 0x000003D1, RZ, !PT, !PT
    [B------:R-:W-:-:S04] @Pt0 IADD3.X Ro1, Pt1, PT, Ro1, 0x00000001, RZ, Pt1, !PT
    [B------:R-:W-:-:S04] @Pt0 IADD3.X Ro2, Pt1, PT, Ro2, 0x00000000, RZ, Pt1, !PT
    [B------:R-:W-:-:S04] @Pt0 IADD3.X Ro3, Pt1, PT, Ro3, 0x00000000, RZ, Pt1, !PT
    [B------:R-:W-:-:S04] @Pt0 IADD3.X Ro4, Pt1, PT, Ro4, 0x00000000, RZ, Pt1, !PT
    [B------:R-:W-:-:S04] @Pt0 IADD3.X Ro5, Pt1, PT, Ro5, 0x00000000, RZ, Pt1, !PT
    [B------:R-:W-:-:S04] @Pt0 IADD3.X Ro6, Pt1, PT, Ro6, 0x00000000, RZ, Pt1, !PT
    [B------:R-:W-:-:S05] @Pt0 IADD3.X Ro7, PT,  PT, Ro7, 0x00000000, RZ, Pt1, !PT
}

FUNCTION NegMod256()
{
    [B------:R-:W-:-:S04]    IADD3.X Rio0, Pt0, PT, RZ, 0xFFFFFC2F, ~Rio0, !PT, PT
    [B------:R-:W-:-:S04]    IADD3.X Rio1, Pt0, PT, RZ, 0xFFFFFFFE, ~Rio1, Pt0, !PT
    [B------:R-:W-:-:S04]    IADD3.X Rio2, Pt0, PT, RZ, 0xFFFFFFFF, ~Rio2, Pt0, !PT
    [B------:R-:W-:-:S04]    IADD3.X Rio3, Pt0, PT, RZ, 0xFFFFFFFF, ~Rio3, Pt0, !PT
    [B------:R-:W-:-:S04]    IADD3.X Rio4, Pt0, PT, RZ, 0xFFFFFFFF, ~Rio4, Pt0, !PT
    [B------:R-:W-:-:S04]    IADD3.X Rio5, Pt0, PT, RZ, 0xFFFFFFFF, ~Rio5, Pt0, !PT
    [B------:R-:W-:-:S04]    IADD3.X Rio6, Pt0, PT, RZ, 0xFFFFFFFF, ~Rio6, Pt0, !PT
    [B------:R-:W-:-:S05]    IADD3.X Rio7, PT, PT, RZ, 0xFFFFFFFF, ~Rio7, Pt0, !PT
}

FUNCTION Copy256()
{
    [B------:R-:W-:-:S01]    IMAD Ro0, RZ, RZ, Ri0
    [B------:R-:W-:-:S01]    MOV Ro1, Ri1
    [B------:R-:W-:-:S01]    IMAD Ro2, RZ, RZ, Ri2
    [B------:R-:W-:-:S01]    MOV Ro3, Ri3
    [B------:R-:W-:-:S01]    IMAD Ro4, RZ, RZ, Ri4
    [B------:R-:W-:-:S01]    MOV Ro5, Ri5
    [B------:R-:W-:-:S01]    IMAD Ro6, RZ, RZ, Ri6
    [B------:R-:W-:-:S05]    MOV Ro7, Ri7
}

FUNCTION Zero256()
{
    [B------:R-:W-:-:S01]    IMAD Ro0, RZ, RZ, RZ
    [B------:R-:W-:-:S01]    MOV Ro1, RZ
    [B------:R-:W-:-:S01]    IMAD Ro2, RZ, RZ, RZ
    [B------:R-:W-:-:S01]    MOV Ro3, RZ
    [B------:R-:W-:-:S01]    IMAD Ro4, RZ, RZ, RZ
    [B------:R-:W-:-:S01]    MOV Ro5, RZ
    [B------:R-:W-:-:S01]    IMAD Ro6, RZ, RZ, RZ
    [B------:R-:W-:-:S06]    MOV Ro7, RZ
}