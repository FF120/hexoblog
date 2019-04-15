---
title: PHP读写XLS
toc: true

tags:
  - PHP
date: 2016-06-23 10:53:27
---
![phpexcel](phpexcel.png)
<!-- more -->

## PHP实现读取和生成XLS文件


```php
 public function read_xls(){
        $file = "D:/TaskHistory/123/template.xls";
        vendor("PHPEXCEL\PHPExcel");
        vendor("PHPEXCEL\PHPExcel\IOFactory");
        $excel = new \PHPExcel();
        date_default_timezone_set('Asia/ShangHai');
        if (!file_exists($file)) {
            exit("not found 31excel5.xls.\n");
        }
        $reader = \PHPExcel_IOFactory::createReader('Excel5'); //设置以Excel5格式(Excel97-2003工作簿)
        $PHPExcel = $reader->load($file); // 载入excel文件
        $sheet = $PHPExcel->getSheet(0); // 读取第一个工作表
        $highestRow = $sheet->getHighestRow(); // 取得总行数
        $highestColumm = $sheet->getHighestColumn(); // 取得总列数

        /** 循环读取每个单元格的数据 */
        for ($row = 1; $row <= $highestRow; $row++){//行数是以第1行开始
            for ($column = 'A'; $column <= $highestColumm; $column++) {//列数是以A列开始
                $value =  $sheet->getCell($column.$row)->getValue();
                $dataset[$row-1][] = $sheet->getCell($column.$row)->getValue();
            }
        }
        dump($dataset);
    }

    public function write_xls()
    {
        $file = "D:/TaskHistory/123/bak.xls";
        vendor("PHPEXCEL\PHPExcel");
        vendor("PHPEXCEL\PHPExcel\IOFactory");
        $product = D('product');
        $data = $product->Relation(true)->select();
        $objPHPExcel = new \PHPExcel();
        /*以下是一些设置 ，什么作者  标题啊之类的*/
        $objPHPExcel->getProperties()->setCreator("转弯的阳光")
            ->setLastModifiedBy("转弯的阳光")
            ->setTitle("数据EXCEL导出")
            ->setSubject("数据EXCEL导出")
            ->setDescription("备份数据")
            ->setKeywords("excel")
            ->setCategory("result file");
            //设置标题
            $objPHPExcel->setActiveSheetIndex(0)
                        ->setCellValue('A' . 1, '品种')
                        ->setCellValue('B' . 1, '材质')
                        ->setCellValue('C' . 1, '规格')
                        ->setCellValue('D' . 1, '宽')
                        ->setCellValue('E' . 1, '长')
                        ->setCellValue('F' . 1, '生产厂家')
                        ->setCellValue('G' . 1, '交货地点')
                        ->setCellValue('H' . 1, '交货仓库')
                        ->setCellValue('I' . 1, '价格（元/吨）')
                        ->setCellValue('G' . 1, '重量（吨）')
                        ->setCellValue('K' . 1, '计重方式')
                        ->setCellValue('L' . 1, '备注');
            foreach ($data as $k => $v) {
                $num = $k + 2;
                $objPHPExcel->setActiveSheetIndex(0)
                    //Excel的第A列，uid是你查出数组的键值，下面以此类推
                    ->setCellValue('A' . $num, $v['product_id'])
                    ->setCellValue('B' . $num, $v['product_name'])
                    ->setCellValue('C' . $num, $v['product_price']);
                //这里添加要加入进去的数据，我没写完
            }

        $objPHPExcel->getActiveSheet()->setTitle('User');
        $objPHPExcel->setActiveSheetIndex(0);
        header('Content-Type: application/vnd.ms-excel');
        header('Content-Disposition: attachment;filename="' . $file . '.xls"');
        header('Cache-Control: max-age=0');
        $objWriter = \PHPExcel_IOFactory::createWriter($objPHPExcel, 'Excel5');
        $objWriter->save('php://output');
    }
```
