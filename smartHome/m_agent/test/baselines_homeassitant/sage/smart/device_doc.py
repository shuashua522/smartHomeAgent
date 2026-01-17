import json
import os

class Device_info_doc:
    # 模拟数据的文件所在目录
    mock_data_dir = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),  # 当前文件所在目录
        'test_mock_data'  # 子目录（无开头斜杠）
    )

    @classmethod
    def load_json(cls, filename):
        file_path = os.path.join(cls.mock_data_dir, filename)
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)

    # 修正：使用类方法延迟加载entities，避免类定义阶段的引用问题
    @classmethod
    @property
    def entities(cls):
        if not hasattr(cls, '_entities'):  # 确保只加载一次
            cls._entities = cls.load_json('entities.json')
        return cls._entities

    @classmethod
    def get_one_liners_string(cls):
        """
        生成格式为 "{entity_id} ({friendly_name}): {capability}" 的字符串，
        各实体信息以换行符分隔，capability 为 entity_id 中 '.' 前面的部分
        """
        result = []
        for entity in cls.entities:  # 引用类属性entities
            entity_id = entity['entity_id']
            friendly_name = entity['attributes']['friendly_name']
            capability = entity_id.split('.')[0]
            result.append(f"{entity_id} ({friendly_name}): {capability}")

        output = '\n'.join(result)
        return output

    @classmethod
    def get_device_capability_string(cls):
        """
        提取所有实体的capability（entity_id中'.'前面的部分），
        去重后转换为以换行符分隔的字符串
        """
        capabilities = set()
        for entity in cls.entities:  # 引用类属性entities
            entity_id = entity['entity_id']
            capability = entity_id.split('.')[0]
            capabilities.add(capability)

        result = '\n'.join(sorted(capabilities))
        return result  # 返回结果而非打印

if __name__ == "__main__":
    print(Device_info_doc.get_one_liners_string())
    print("======================")
    print(Device_info_doc.get_device_capability_string())