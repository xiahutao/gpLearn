# -*- coding: utf-8 -*-
'''
@Time    : 2020/10/27 14:40
@Author  : zhangfang
@File    : png_pdf.py
'''
from gpLearn.xt.strategy_state import *

if __name__ == "__main__":
    resualt_path = 'c:/e/hmm/'
    document = Document()
    head = document.add_heading(0)
    run = head.add_run(f'HMM模型在商品期货市场特征因子分析-以RB为例')
    run.font.name = u'黑体'  # 设置字体为黑体
    run._element.rPr.rFonts.set(qn('w:eastAsia'), u'黑体')
    run.font.size = Pt(24)  # 设置大小为24磅
    run.font.color.rgb = RGBColor(0, 0, 0)  # 设置颜色为黑色
    head.alignment = WD_PARAGRAPH_ALIGNMENT.CENTER  # 居中
    factor_list = ['close', 'volume', 'dist_10_5', 'dist_15_5', 'dist_20_5', 'dist_15_10', 'dist_20_10', 'dist_20_15',
                   'dist_30_15', 'log_return', 'log_return_5', 'MACD', 'dist_high_10', 'dist_high_15', 'dist_high_20',
                   'dist_high_25', 'dist_low_10', 'dist_low_15', 'dist_low_20', 'dist_low_25', 'MA_5', 'MA_10', 'MA_15',
                   'MA_20', 'MA_25', 'MA_30']
    for i in factor_list:
        document.add_picture(f'{resualt_path}/fig/{i}_cum_ret.png', width=Inches(6.0))
        document.add_picture(f'{resualt_path}/fig/{i}.png', width=Inches(6.0))
    document.save(f'{resualt_path}/report/HMM模型在商品期货市场特征因子分析-以RB为例.docx')

    FormatConvert.word_to_pdf(f'{resualt_path}/report/HMM模型在商品期货市场特征因子分析-以RB为例.docx',
                              f'{resualt_path}/report/HMM模型在商品期货市场特征因子分析-以RB为例.pdf')
