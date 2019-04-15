---
title: Lavarel 后台组件frozenode的使用
toc: true

tags:
  - laravel
  - frozenode
date: 2016-06-11 20:01:45
---



## 安装
## 使用
<!-- more -->
### 导航栏配置
administrator.php:
uri 后台管理地址
model_config_path 模型地址
setting_config_path 每个子菜单配置文件的地址
menu 导航栏菜单项目
```php
<?php

return array(

	/**
	 * Package URI
	 *
	 * @type string
	 */
	'uri' => 'eadmin',

	/**
	 * Page title
	 *
	 * @type string
	 */
	'title' => 'PomeMartAdminPanel',

	/**
	 * The path to your model config directory
	 *
	 * @type string
	 */
	'model_config_path' => app('path') . '/config/administrator',

	/**
	 * The path to your settings config directory
	 *
	 * @type string
	 */
	'settings_config_path' => app('path') . '/config/administrator/settings',

	/**
	 * The menu structure of the site. For models, you should either supply the name of a model config file or an array of names of model config
	 * files. The same applies to settings config files, except you must prepend 'settings.' to the settings config file name. You can also add
	 * custom pages by prepending a view path with 'page.'. By providing an array of names, you can group certain models or settings pages
	 * together. Each name needs to either have a config file in your model config path, settings config path with the same name, or a path to a
	 * fully-qualified Laravel view. So 'users' would require a 'users.php' file in your model config path, 'settings.site' would require a
	 * 'site.php' file in your settings config path, and 'page.foo.test' would require a 'test.php' or 'test.blade.php' file in a 'foo' directory
	 * inside your view directory.
	 *
	 * @type array
	 *
	 * 	array(
	 *		'E-Commerce' => array('collections', 'products', 'product_images', 'orders'),
	 *		'homepage_sliders',
	 *		'users',
	 *		'roles',
	 *		'colors',
	 *		'Settings' => array('settings.site', 'settings.ecommerce', 'settings.social'),
	 * 		'Analytics' => array('E-Commerce' => 'page.ecommerce.analytics'),
	 *	)
	 */
	'menu' => array
	(
		'商品' => array
		(
			'products' => 'products',
			'product images' => 'productImages',
			'tags' => 'tags',
			'attributes' => 'attributes',
			'categories' => 'categories',
            'productDetailImage' => 'productDetailImage',
            'accounting' => 'accounting',
            'accounting_withdraw' => 'accounting_withdraw'
		),
		'店铺' => array(
			 'stores' => 'stores',
			 'cmsItems' =>'cmsItems',
			 'storePromos' => 'storePromos',
			 'storeOuterlinks' => 'storeOuterlinks',
			 'shares' => 'shares',
             'themes' => 'themes'
		),
		'优惠券' => array(
			'coupons' => 'coupons',
			'couponCreationRules' => 'couponCreationRules',
			'fixedDiscountCoupons' => 'fixedDiscountCoupons',
			'percentageDiscountCoupons' => 'percentageDiscountCoupons',
			'amountOffOverCoupons' => 'amountOffOverCoupons'
		),
		'订单' => array(
			'orders' => 'orders',
			'shipments' => 'shipments',
			'trackings' => 'trackings'
		),
		'评价' => array(
			'ratings'=>'ratings',
			'productRatings' => 'productRatings'
		)
	),

	/**
	 * The permission option is the highest-level authentication check that lets you define a closure that should return true if the current user
	 * is allowed to view the admin section. Any "falsey" response will send the user back to the 'login_path' defined below.
	 *
	 * @type closure
	 */
	'permission'=> function()
	{
		if (Auth::check() && Auth::user()->hasRole('Admin'))
		{
		    return true;
		}
		return false;
	},

	/**
	 * This determines if you will have a dashboard (whose view you provide in the dashboard_view option) or a non-dashboard home
	 * page (whose menu item you provide in the home_page option)
	 *
	 * @type bool
	 */
	'use_dashboard' => false,

	/**
	 * If you want to create a dashboard view, provide the view string here.
	 *
	 * @type string
	 */
	'dashboard_view' => '',

	/**
	 * The menu item that should be used as the default landing page of the administrative section
	 *
	 * @type string
	 */
	'home_page' => 'products',

	/**
	 * The route to which the user will be taken when they click the "back to site" button
	 *
	 * @type string
	 */
	'back_to_site_path' => '/',

	/**
	 * The login path is the path where Administrator will send the user if they fail a permission check
	 *
	 * @type string
	 */
	'login_path' => 'accounts/signin',

	/**
	 * The logout path is the path where Administrator will send the user when they click the logout link
	 *
	 * @type string
	 */
	'logout_path' => 'accounts/signout',

	/**
	 * This is the key of the return path that is sent with the redirection to your login_action. Session::get('redirect') will hold the return URL.
	 *
	 * @type string
	 */
	'login_redirect_key' => 'redirect',

	/**
	 * Global default rows per page
	 *
	 * @type NULL|int
	 */
	'global_rows_per_page' => 20,

	/**
	 * An array of available locale strings. This determines which locales are available in the languages menu at the top right of the Administrator
	 * interface.
	 *
	 * @type array
	 */
	'locales' => array('zh-CN','en'),

);
```
accounting.php :
```php
<?php

/**
 *  model config
 */

return array(

	'title' => '财务',

	'single' => '财务',

	'model' => 'PomeMartDomainModel\Entities\Accounting',

	'columns' => array(
		'id',
		'store_id' => array (
			'title' => '店铺ID',
		),
		'alipay_account' => array(
			'title' => '支付宝账户',
		),
		'total_income' => array(
			'title' => '总收入'
		),
		'balance' => array(
			'title' => '待提金额'
		),
		'last_order_received_on' => array(
			'title' => '上次提取现金的时间',
		)
	),

	'edit_fields' => array(
        'store_id' => array(
            'title' => '店铺ID'
        ),
        'alipay_account' => array(
            'title' => '支付宝账户',
        ),
        'total_income' => array(
            'title' => '总收入',
            'type' => 'number'
        ),
        'balance' => array(
            'title' => '待提取金额'
        ),
        'last_order_received_on' => array(
            'title' => '上次提取现金的时间',
            'type' => 'datetime'

        )


	),

	'filters' => array(
        'store_id' => array(
            'title' => '店铺ID'
        ),
        'alipay_account' => array(
            'title' => '支付宝账户'
        )
	),
);
```
里面的model 指定要操作的数据库的表的对应的模型的位置。

参考文档地址：[http://administrator.frozennode.com/](http://administrator.frozennode.com/)

## 添加按钮执行自定义代码
执行全局操作，没有输入参数
``` php
 'global_actions' => array(
        //Create Excel Download
        'clear_cache' => array(
            'title' => 'Clear Cache',
            'messages' => array(
                'active' => 'clear cache ...',
                'success' => 'success!',
                'error' => 'failed!',
            ),
            //the Eloquent query builder is passed to the closure
            'action' => function($query)
                {
                    $store = Auth::user()->store;
                    Cache::forget('store_'.$store->store_alias);
                    return  true;
                }
        ),
    ),
```
针对每条数据，执行自定义操作，传入参数$data就是编辑后的数据
``` php
 'actions' => array(
        //Clearing the site cache
        'save_color' => array(
            'title' => 'Save Color',
            'messages' => array(
                'active' => 'Clearing cache...',
                'success' => 'Cache cleared!',
                'error' => 'There was an error while clearing the cache',
            ),
            //the settings data is passed to the function and saved if a truthy response is returned
            'action' => function(&$data)
                {

                    //return true to flash the success message
                    //return false to flash the default error
                    //return a string to show a custom error
                    //return a Response::download() to initiate a file download
                    return "$data";
                }
        ),
    ),
```
====
