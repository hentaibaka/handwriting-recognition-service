from celery import shared_task
from recognition_module.recognition_module import RecognitionModule
from django.apps import apps


@shared_task()
def start_train(train_id):
    Train = apps.get_model('ai_service', 'Train')
    AIModel = apps.get_model('ai_service', 'AIModel')
    train = Train.objects.get(pk=train_id)

    if train and train.model_to_train and train.train_set:  # type: ignore
        dataset_name = f'dataset-{train.pk}'
        model_name_train = train.model_to_train.name # type: ignore
        model_name_trained = f'model-{train.model_to_train.pk}-{dataset_name}' # type: ignore

        strings = (string for string in train.train_set.strings.all() # type: ignore
                   if string and string.page and string.page.image and string.coords) 

        if strings:
            data = ((string.page.image.path, string.coords, string.text) 
                    for string in strings)
            
            new_prediction = False

            status, msg = RecognitionModule.train(list(data), dataset_name, model_name_train, train.model_to_train.model_type, model_name_trained, # type: ignore
                                                  train.num_iter, train.val_interval, train.batch_size, new_prediction) # type: ignore

            if status:
                train.status = train.StatusChoices.DONE # type: ignore
                trained_model = AIModel.objects.create(name=model_name_trained,
                                                       is_current=False, 
                                                       model_type=train.model_to_train.model_type) # type: ignore
                train.trained_model = trained_model # type: ignore
            else:
                train.status = train.StatusChoices.ERROR # type: ignore
        else:
            train.status = train.StatusChoices.ERROR # type: ignore
            msg = "Датасет не должен быть пустым"
    else:
        train.status = train.StatusChoices.ERROR # type: ignore
        msg = "Отсутствует датасет или модель, которую нужно обучить"

    train.message = msg # type: ignore
    train.save()
    