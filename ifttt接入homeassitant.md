

## 流程

> 教程地址

[IFTTT - Home Assistant --- IFTTT - Home Assistant](https://www.home-assistant.io/integrations/ifttt)



### ifttt端

http://62.234.0.27:8123/api/webhook/c4d2aaab1a83b35bef069b7941de089b01043a0018cd8c7c38fe042ad7b41d93

|    配置项    |                           填写内容                           |
| :----------: | :----------------------------------------------------------: |
|     URL      | `http://172.17.0.3:8123/api/webhook/c4d2aaab1a83b35bef069b7941de089b01043a0018cd8c7c38fe042ad7b41d93` |
|    Method    |                            `POST`                            |
| Content Type |                      `application/json`                      |
|     Body     | `{ "action": "call_service", "service": "light.turn_on", "entity_id": "light.yeelink_cn_1162511951_mbulb3_s_2" }` |

### homeassitant端 配置

> 灯的实体ID：
>
> light.yeelink_cn_1162511951_mbulb3_s_2

```yaml
automation:
- alias: "The optional automation alias"
  triggers:
    - trigger: event
      event_type: ifttt_webhook_received
      event_data:
        action: call_service  # the same action 'name' you used in the Body section of the IFTTT recipe
  actions:
    - action: '{{ trigger.event.data.service }}'
      target:
        entity_id: '{{ trigger.event.data.entity_id }}'
```

##### 新手友好：如何把这个 YAML 配置到 HA 里？

1. 打开 HA 后台，点击左侧「设置」→「自动化与场景」→「创建自动化」→「切换到 YAML 模式」（右上角有个「编辑 YAML」的按钮）。

2. 清空原有内容，把上面的 YAML 复制进去，然后做 3 处修改：

   - `alias`：改成自定义名称，比如`"IFTTT 触发打开客厅灯"`。
   - `entity_id`：把 `light.living_room` 换成你自己 HA 里实际的灯的实体 ID（比如之前的`light.yeelink_cn_1162511951_mbulb3_s_2`）。
   - 确认 `action: call_service` 和 IFTTT JSON 里的 `action` 值一致。

   

3. 点击「保存」，至此 HA 自动化就配置完成了。