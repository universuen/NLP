from Utilities import DataLoader, Models

if __name__ == '__main__':
    # 加载数据
    dataset = DataLoader('./dataset/')
    dataset.load()
    # 初始化模型
    models = Models(
            max_seq_len=dataset.max_seq_len,
            vocabulary_size=dataset.vocabulary_size,
            embedding_size=dataset.embedding_size,
            embedding_matrix=dataset.embedding_matrix
    )
    # 编译模型
    print("Compiling models")
    models.compile()
    # 训练模型
    print("Training model A")
    models.train(
            model_name='a',
            training_x=dataset.training_a.x,
            training_labels=dataset.training_a.labels,
            validation_x=dataset.validation_a.x,
            validation_labels=dataset.validation_a.labels,
            # epochs=10
    )
    print("Training model B")
    models.train(
            model_name='b',
            training_x=dataset.training_b.x,
            training_labels=dataset.training_b.labels,
            validation_x=dataset.validation_b.x,
            validation_labels=dataset.validation_b.labels,
            # epochs=3
    )
    print("Training model C")
    models.train(
            model_name='c',
            training_x=dataset.training_c.x,
            training_labels=dataset.training_c.labels,
            validation_x=dataset.validation_c.x,
            validation_labels=dataset.validation_c.labels,
            # epochs=2
    )
    # 测试模型
    models.evaluate(
            model_name='a',
            x=dataset.test_a.x,
            labels=dataset.test_a.labels
    )
    models.evaluate(
            model_name='b',
            x=dataset.test_b.x,
            labels=dataset.test_b.labels
    )
    models.evaluate(
            model_name='c',
            x=dataset.test_c.x,
            labels=dataset.test_c.labels
    )
