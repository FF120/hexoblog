---
title: laravel 4 note 01
date: 2016-06-11 13:58:44
toc: true

tags:
- laravel
---
# artisan常用命令
- `php artisan generate:seed page` 新建`app/database/seeds/PageTableSeeder.php`文件，seed文件可以用来随机生成数据填充到数据库。
- `php artisan generate:view admin._layout.default` 新建`app/views/admin/_layout/default.blade.php`文件，这是模版文件。
- `php artisan generate:mode article` 新建`app/models/Article.php`文件，这是模型，与数据库操作相关。
- `php artisan serve` 开启Laravel自带的Web服务器，使用地址`localhost:8000`访问
- `php artisan migrate`   执行数据迁移,该命令会将migration下面改动的文件执行一遍，确保和数据库的一致。


<!-- more -->


# 使用Composer安装需要的依赖
1. 在`composer.json`的`require`或`require-dev`添加组件名称和版本号
```
 "edvinaskrucas/notification": "3.0.1"
```
2. 在项目的根目录下执行`composer update`命令
3. 在`app.php`中的`providers`添加一行
```
 'Krucas\Notification\NotificationServiceProvider',
```
4. 在`aliases`中添加一行
```
  'Notification'      => 'Krucas\Notification\Facades\Notification',
```

# 常见错误
1. 新增的类编译的时候提示找不到该类：执行`composer auto-dump`
2. 新建文件夹后找不到文件：在`composer.json`中的`autoload`的`classmap`新增一行该文件夹的路径。

# 常用方法
1. `Sentry::check()` 检查用户是否登陆
2. `Input::get('email')` 获取前台通过`get`方式提交的`email`字段
3. `Redirect::route('admin.login')`
4. `Redirect::to('eadmin/products')`
5. `HTML::link('account/newaccount','register',array('class'=>'default-btn'))` Blade模版语法
6. `Form::open(array('url'=>'accounts/sigin'))`

# Eloqyent操作

``` php
//建立一个Page模型，与数据库中的表user关联
class Page extends \Eloquent{
	protected $table='users'; //指定使用的数据库，不指定的时候默认为类名称的小写复数形式
	protected $primaryKey='id'; //指定主键的名称，不指定默认为id
	protected $timestamps = false; //取消数据库的created_at 和 updated_at两个字段，默认情况下每个数据库都有这两个字段，是框架自动生成和维护的。
	protected $appends = array('a1','a2'); //需要使用但是数据库中没有定义的字段
	protected $hidden = array('h1','h2'); //需要隐藏的字段，隐藏的字段不会出现在查询结果中。
	use SoftDeletingTrait; //开启软删除功能，默认情况下该功能是不打开的。开启软删除之后，删除数据库的命令不会真正执行，而是更新数据库的deleted_at字段。
	//如果要自定义返回的时间戳的格式，可以改写此方法实现
	protected function getDateFormat(){
		return 'U';
	}
}
```
